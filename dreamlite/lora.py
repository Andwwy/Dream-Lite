"""LoRA written from scratch: layer, training loop, adapter save, merge into base weights.

Used unchanged by B2 (gold solutions), B3 (self-labelled real questions) and dreaming
(self-labelled invented questions), so the three differ only in their training data.
"""
import math
import random
import time

import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, with W frozen. B starts at zero, so training starts at W."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(model, rank=16, alpha=32):
    """Freeze the model and wrap every target linear layer. Returns {module path: LoRALinear}."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = {}
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in TARGETS and isinstance(child, nn.Linear):
                lora = LoRALinear(child, rank, alpha)
                setattr(mod, child_name, lora)
                wrapped[f"{name}.{child_name}"] = lora
    return wrapped


def encode(tok, problem, solution, max_len=1024):
    """Token ids and labels; the loss covers the solution and the end-of-text token only."""
    q = tok(f"Problem:\n{problem}\n\nSolution:\n")["input_ids"]
    s = tok(solution + tok.eos_token)["input_ids"]
    ids = (q + s)[:max_len]
    return ids, [-100] * min(len(q), len(ids)) + ids[len(q):]


def make_batches(data, batch_tokens):
    """Batches of similar length, capped by padded token count."""
    data = sorted(data, key=lambda d: len(d[0]))
    batches, cur = [], []
    for d in data:
        if cur and (len(cur) + 1) * len(d[0]) > batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(d)
    return batches + [cur]


def train(model, tok, wrapped, pairs, *, seed, lr=1e-4, batch_tokens=3000, warmup=30,
          epochs=None, budget_tokens=None, on_epoch=None, log_every=50):
    """Train the adapters on (problem, solution) pairs.

    Stops after `epochs` passes or once `budget_tokens` training tokens have been seen,
    whichever is given. Returns a dict of statistics.
    """
    assert (epochs is None) != (budget_tokens is None)
    torch.manual_seed(seed)
    for l in wrapped.values():                       # re-initialise with this seed
        nn.init.kaiming_uniform_(l.A, a=math.sqrt(5))
        nn.init.zeros_(l.B)
    params = [p for l in wrapped.values() for p in (l.A, l.B)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    head = model.get_output_embeddings()
    data = [encode(tok, p["problem"], p["solution"]) for p in pairs]
    batches = make_batches(data, batch_tokens)
    epoch_tokens = sum(len(ids) for ids, _ in data)
    rng = random.Random(seed)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    model.train()
    step, seen, losses, t0, epoch = 0, 0, [], time.time(), 0
    done = False
    while not done:
        rng.shuffle(batches)
        for batch in batches:
            L = max(len(ids) for ids, _ in batch)
            x = torch.tensor([ids + [pad] * (L - len(ids)) for ids, _ in batch]).cuda()
            y = torch.tensor([lab + [-100] * (L - len(lab)) for _, lab in batch]).cuda()
            mask = torch.tensor([[1] * len(ids) + [0] * (L - len(ids)) for ids, _ in batch]).cuda()
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / warmup)
            # Logits only where there is a label: the vocabulary is 152k wide, so logits for
            # prompt and padding positions would dominate GPU memory.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.base_model(input_ids=x, attention_mask=mask).last_hidden_state[:, :-1]
                tgt = y[:, 1:]
                sel = tgt != -100
                loss = nn.functional.cross_entropy(head(h[sel]).float(), tgt[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            seen += int(mask.sum())
            losses.append(loss.item())
            if step % log_every == 0:
                print(f"step {step} epoch {epoch} tokens {seen} loss {sum(losses[-log_every:]) / log_every:.4f} "
                      f"{seen / (time.time() - t0):.0f} tok/s", flush=True)
            if budget_tokens is not None and seen >= budget_tokens:
                done = True
                break
        epoch += 1
        if on_epoch and not done:
            on_epoch(epoch)
        if epochs is not None and epoch >= epochs:
            done = True
    k = max(1, len(losses) // 10)
    return dict(steps=step, train_tokens=seen, epoch_tokens=epoch_tokens, examples=len(pairs),
                epochs_done=round(seen / epoch_tokens, 2), seconds=round(time.time() - t0, 1),
                tokens_per_s=round(seen / (time.time() - t0)), loss_first=round(sum(losses[:k]) / k, 4),
                loss_last=round(sum(losses[-k:]) / k, 4), lr=lr, batch_tokens=batch_tokens,
                trainable_params=sum(p.numel() for p in params),
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))


def save_adapter(wrapped, path, rank, alpha):
    torch.save(dict(rank=rank, alpha=alpha,
                    weights={n: (l.A.detach().float().cpu(), l.B.detach().float().cpu())
                             for n, l in wrapped.items()}), path)


def merge(base_name, adapter_path, out_dir):
    """Write a plain model with W + (alpha / r) B A, loadable by vLLM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ad = torch.load(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16)
    mods = dict(model.named_modules())
    scale = ad["alpha"] / ad["rank"]
    for name, (A, B) in ad["weights"].items():
        w = mods[name].weight
        w.data = (w.data.float() + (B @ A) * scale).to(w.dtype)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_name).save_pretrained(out_dir)
