"""Smoke test for one small Dream-Lite night: how long does each stage take?

Stages (each runs as its own process, because vLLM reserves the whole GPU):
  gen     imagine N problems, rehearse k solutions each, majority-vote filter
  train   LoRA on the kept pairs, merge into the base weights, save
  eval    greedy 4-shot accuracy on MATH-500 for one model
  report  print the timings and scale them to a 4,000-problem night

This is timing code. The prompts, the answer checker and the LoRA settings are
simple stand-ins for the team's real pipeline, so treat the accuracy numbers as
a sanity check only.
"""
import argparse
import json
import random
import re
import signal
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

T0 = time.time()
BASE = "Qwen/Qwen3-1.7B-Base"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
STOP = ["\n\nProblem:", "\nProblem:"]
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------- data and prompts

def load_math_train():
    from datasets import load_dataset
    rows = []
    for s in SUBJECTS:
        rows += list(load_dataset("EleutherAI/hendrycks_math", s, split="train"))
    return rows


def fewshot_prefix(train):
    """Four fixed short worked examples, one per subject, as the chain-of-thought prompt."""
    rng = random.Random(0)
    shots, seen = [], set()
    pool = [r for r in train if len(r["solution"]) < 350 and r["solution"].count("\\boxed") == 1
            and "[asy]" not in r["problem"] + r["solution"]]
    rng.shuffle(pool)
    for r in pool:
        if r["type"] not in seen:
            seen.add(r["type"])
            shots.append(r)
        if len(shots) == 4:
            break
    return "".join(f"Problem:\n{r['problem']}\n\nSolution:\n{r['solution']}\n\n" for r in shots)


def solve_prompt(prefix, problem):
    return f"{prefix}Problem:\n{problem}\n\nSolution:\n"


# ---------------------------------------------------------------- answers and voting

def last_boxed(text):
    i = text.rfind("\\boxed")
    j = text.find("{", i) if i >= 0 else -1
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalize(a):
    if a is None:
        return None
    a = a.strip().replace("\n", "")
    for s in ("\\!", "\\,", "\\ ", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\$", "$", "\\%", "%"):
        a = a.replace(s, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    no_text = re.sub(r"\\(text|mbox)\{[^}]*\}", "", a)
    a = no_text if no_text.strip() else re.sub(r"\\(text|mbox)\{([^}]*)\}", r"\2", a)
    a = a.replace(" ", "").rstrip(".")
    a = re.sub(r"^[a-zA-Z]=", "", a)
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", a):
        a = a.replace(",", "")
    return a or None


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()


@lru_cache(maxsize=None)
def _parse(a):
    from sympy.parsing.latex import parse_latex
    try:
        return parse_latex(a)
    except _Timeout:
        raise
    except Exception:
        return None


@lru_cache(maxsize=None)
def equiv(a, b):
    """String match after normalising, then a sympy check with a 2 s limit."""
    if a == b:
        return True
    import sympy
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        x, y = _parse(a), _parse(b)
        if x is None or y is None:
            return False
        return bool(sympy.simplify(x - y) == 0)
    except (_Timeout, Exception):
        return False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def vote(answers):
    """Group answers by equivalence. Returns (majority answer, indices that gave it)."""
    groups = []
    for i, a in enumerate(answers):
        if a is None:
            continue
        for g in groups:
            if equiv(g[0], a):
                g[1].append(i)
                break
        else:
            groups.append((a, [i]))
    if not groups:
        return None, []
    return max(groups, key=lambda g: len(g[1]))


# ---------------------------------------------------------------- bookkeeping

def record(out, stage, **kw):
    f = Path(out) / "timings.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[stage] = kw
    f.write_text(json.dumps(d, indent=2))
    print(f"[{stage}]", json.dumps(kw))


def load_llm(model, max_num_seqs):
    from vllm import LLM
    t = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, seed=0,
              gpu_memory_utilization=0.90, max_num_seqs=max_num_seqs)
    return llm, time.time() - t


def run(llm, prompts, **sp):
    from vllm import SamplingParams
    t = time.time()
    outs = llm.generate(prompts, SamplingParams(**sp))
    dt = time.time() - t
    comps = [c for o in outs for c in o.outputs]
    n_out = sum(len(c.token_ids) for c in comps)
    stats = dict(seconds=round(dt, 1), sequences=len(comps), output_tokens=n_out,
                 prompt_tokens=sum(len(o.prompt_token_ids) for o in outs),
                 output_tokens_per_s=round(n_out / dt), mean_output_tokens=round(n_out / len(comps), 1),
                 hit_max_tokens=round(sum(c.finish_reason == "length" for c in comps) / len(comps), 3))
    return outs, stats


# ---------------------------------------------------------------- stage: gen

def stage_gen(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_math_train()
    prefix = fewshot_prefix(train)
    llm, load_s = load_llm(BASE, a.max_num_seqs)
    record(out, "gen_load", seconds=round(load_s, 1))

    # Imagine: 4 real train questions as style seeds, the model writes a fifth.
    rng = random.Random(0)
    n_prompts = int(a.n * 1.3)
    prompts = []
    for _ in range(n_prompts):
        seeds = rng.sample(train, 4)
        prompts.append("Below are problems from a mathematics competition.\n\n"
                       + "".join(f"Problem:\n{s['problem']}\n\n" for s in seeds) + "Problem:\n")
    outs, st = run(llm, prompts, temperature=1.0, top_p=0.95, max_tokens=300,
                   stop=["\n\nProblem:", "\nProblem:", "\n\nSolution", "\nSolution"], seed=0)
    real = {r["problem"].strip() for r in train}
    problems, seen = [], set()
    for o in outs:
        c = o.outputs[0]
        p = c.text.strip()
        if c.finish_reason == "stop" and 20 <= len(p) <= 1000 and p not in seen and p not in real:
            seen.add(p)
            problems.append(p)
    st["prompts"], st["valid_problems"] = n_prompts, len(problems)
    problems = problems[:a.n]
    record(out, "imagine", **st)

    # Rehearse: k sampled solutions per dreamed problem.
    outs, st = run(llm, [solve_prompt(prefix, p) for p in problems], n=a.k, temperature=0.8,
                   max_tokens=a.max_tokens, stop=STOP, seed=0)
    st["problems"] = len(problems)
    record(out, "rehearse", **st)

    # Filter: keep 5 <= agreement <= 7 of 8, and up to 2 solutions with the majority answer.
    t = time.time()
    hist, kept, rows = Counter(), [], []
    lo, hi = a.k // 2 + 1, a.k - 1
    for p, o in zip(problems, outs):
        sols = [c.text.strip() for c in o.outputs]
        answers = [normalize(last_boxed(s)) for s in sols]
        maj, idx = vote(answers)
        hist[len(idx)] += 1
        keep = lo <= len(idx) <= hi
        rows.append(dict(problem=p, majority=maj, agree=len(idx), kept=keep, answers=answers,
                         solutions=[sols[i] for i in idx[:2]]))
        if keep:
            kept += [dict(problem=p, solution=sols[i]) for i in idx[:2]]
    (out / "dreamed.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (out / "kept.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    record(out, "filter", seconds=round(time.time() - t, 1), problems=len(problems),
           kept_problems=sum(r["kept"] for r in rows), kept_pairs=len(kept),
           no_boxed_answer=round(sum(x is None for r in rows for x in r["answers"]) / (len(rows) * a.k), 3),
           unanimous=hist[a.k], no_majority=sum(v for c, v in hist.items() if c < lo),
           agreement_histogram={str(c): hist[c] for c in range(a.k + 1)})
    record(out, "gen_total", seconds=round(time.time() - T0, 1))


# ---------------------------------------------------------------- stage: train

def stage_train(a):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(a.out)
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    t = time.time()
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
    load_s = time.time() - t

    class LoRALinear(nn.Module):
        def __init__(self, base, r, alpha):
            super().__init__()
            self.base, self.scale = base, alpha / r
            self.A = nn.Parameter(torch.empty(r, base.in_features, device="cuda"))
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device="cuda"))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        def forward(self, x):
            return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

        def merged(self):
            delta = (self.B @ self.A) * self.scale
            self.base.weight.data += delta.to(self.base.weight.dtype)
            return self.base

    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in LORA_TARGETS and isinstance(child, nn.Linear):
                lora = LoRALinear(child, a.rank, 2 * a.rank)
                setattr(mod, child_name, lora)
                wrapped.append((mod, child_name, lora))
    params = [p for _, _, l in wrapped for p in (l.A, l.B)]

    # Training pairs. If the filter kept too few for a stable timing, top up with other
    # majority-vote solutions (flagged in the output; timing only).
    pairs = [json.loads(l) for l in (out / "kept.jsonl").read_text().splitlines() if l]
    n_kept = len(pairs)
    if len(pairs) < a.min_examples:
        extra = [dict(problem=r["problem"], solution=s)
                 for r in map(json.loads, (out / "dreamed.jsonl").read_text().splitlines())
                 if not r["kept"] and r["agree"] > 4 for s in r["solutions"]]
        pairs += extra[:a.min_examples - len(pairs)]
    random.Random(0).shuffle(pairs)

    def encode(ex):
        q = tok(f"Problem:\n{ex['problem']}\n\nSolution:\n")["input_ids"]
        s = tok(ex["solution"] + tok.eos_token)["input_ids"]
        ids = (q + s)[:a.max_len]
        return ids, [-100] * min(len(q), len(ids)) + ids[len(q):]

    # Batches of similar length, capped by padded token count, so one long solution
    # cannot blow up GPU memory.
    data = sorted((encode(ex) for ex in pairs), key=lambda d: len(d[0]))
    batches, cur = [], []
    for d in data:
        if cur and (len(cur) + 1) * len(d[0]) > a.batch_tokens:
            batches.append(cur)
            cur = []
        cur.append(d)
    batches.append(cur)
    random.Random(0).shuffle(batches)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    head = model.get_output_embeddings()
    model.train()
    losses, n_tok = [], 0
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for batch in batches:
        L = max(len(ids) for ids, _ in batch)
        x = torch.tensor([ids + [tok.pad_token_id or 0] * (L - len(ids)) for ids, _ in batch]).cuda()
        y = torch.tensor([lab + [-100] * (L - len(lab)) for _, lab in batch]).cuda()
        mask = torch.tensor([[1] * len(ids) + [0] * (L - len(ids)) for ids, _ in batch]).cuda()
        # Logits only where there is a label: the vocabulary is 152k wide, so logits for
        # prompt and padding positions would dominate GPU memory.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.base_model(input_ids=x, attention_mask=mask).last_hidden_state[:, :-1]
            tgt = y[:, 1:]
            sel = tgt != -100
            loss = nn.functional.cross_entropy(head(h[sel]).float(), tgt[sel])
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        n_tok += int(mask.sum())
    train_s = time.time() - t

    # Merge the adapter into the base weights and save a plain model for vLLM.
    t = time.time()
    torch.save({f"{i}.{n}": getattr(l, n).detach().cpu() for i, (_, _, l) in enumerate(wrapped)
                for n in ("A", "B")}, out / "lora_adapter.pt")
    for mod, child_name, lora in wrapped:
        setattr(mod, child_name, lora.merged())
    model.save_pretrained(a.merged)
    tok.save_pretrained(a.merged)
    merge_s = time.time() - t

    k = max(1, len(losses) // 5)
    record(out, "train", load_seconds=round(load_s, 1), seconds=round(train_s, 1),
           merge_save_seconds=round(merge_s, 1), examples=len(pairs), kept_pairs=n_kept,
           timing_only_topup=len(pairs) - n_kept, steps=len(losses), tokens=n_tok,
           tokens_per_s=round(n_tok / train_s), examples_per_s=round(len(pairs) / train_s, 1),
           loss_first=round(sum(losses[:k]) / k, 3), loss_last=round(sum(losses[-k:]) / k, 3),
           lora_modules=len(wrapped), trainable_params=sum(p.numel() for p in params),
           rank=a.rank, lr=a.lr, batch_tokens=a.batch_tokens,
           peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1),
           total_seconds=round(time.time() - T0, 1))


# ---------------------------------------------------------------- stage: eval

def stage_eval(a):
    from datasets import load_dataset
    out = Path(a.out)
    prefix = fewshot_prefix(load_math_train())
    test = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    llm, load_s = load_llm(a.model, a.max_num_seqs)
    outs, st = run(llm, [solve_prompt(prefix, r["problem"]) for r in test],
                   temperature=0.0, max_tokens=a.max_tokens, stop=STOP)
    t = time.time()
    correct, no_answer = 0, 0
    for r, o in zip(test, outs):
        pred = normalize(last_boxed(o.outputs[0].text))
        no_answer += pred is None
        correct += pred is not None and equiv(pred, normalize(r["answer"]))
    record(out, f"eval_{a.tag}", load_seconds=round(load_s, 1), score_seconds=round(time.time() - t, 1),
           accuracy=round(100 * correct / len(test), 1), n=len(test),
           no_boxed_answer=no_answer, total_seconds=round(time.time() - T0, 1), **st)


# ---------------------------------------------------------------- stage: report

def stage_report(a):
    d = json.loads((Path(a.out) / "timings.json").read_text())
    g, im, re_, fi, tr = d["gen_load"], d["imagine"], d["rehearse"], d["filter"], d["train"]
    scale = a.night / re_["problems"]
    pairs = fi["kept_pairs"] * scale
    ev = d.get("eval_merged") or d["eval_base"]
    night = dict(
        load_vllm=g["seconds"],
        imagine=im["seconds"] * scale,
        rehearse=re_["seconds"] * scale,
        filter=fi["seconds"] * scale,
        consolidate=tr["load_seconds"] + pairs / tr["examples_per_s"] + tr["merge_save_seconds"],
        eval_math500=ev["total_seconds"],
    )
    print(json.dumps(d, indent=2))
    print(f"\nScaled to {a.night} problems x {re_['sequences'] // re_['problems']} samples (seconds):")
    for k, v in night.items():
        print(f"  {k:14s}{v:8.0f}")
    print(f"  {'total':14s}{sum(night.values()):8.0f}  = {sum(night.values()) / 3600:.2f} GPU-hours")
    print(f"  kept pairs per night ~ {pairs:.0f} (replay data, GSM8K and MMLU evaluation not included)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "train", "eval", "report"])
    ap.add_argument("--out", default="/workspace/runs/smoke")
    ap.add_argument("--n", type=int, default=200, help="dreamed problems")
    ap.add_argument("--k", type=int, default=8, help="solutions per problem")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--merged", default="/root/smoke_merged")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_tokens", type=int, default=4000, help="padded tokens per batch")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--min_examples", type=int, default=256)
    ap.add_argument("--night", type=int, default=4000, help="problems per full night, for scaling")
    a = ap.parse_args()
    {"gen": stage_gen, "train": stage_train, "eval": stage_eval, "report": stage_report}[a.stage](a)
