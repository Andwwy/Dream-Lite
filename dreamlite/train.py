"""Train a LoRA adapter on (problem, solution) pairs.

  B2, gold solutions:    python -m dreamlite.train --data gold --tag b2_s0 --seed 0 --epochs 3 --save_epochs
  B3, self-labelled:     python -m dreamlite.train --data runs/b3/b3_s0.kept.jsonl --tag b3_s0 --seed 0 --budget_tokens N

--budget_tokens is the fair-comparison rule: every method trains on the same number of tokens.
The adapter goes to <out>/adapters/<tag>.pt (and <tag>.ep<N>.pt with --save_epochs).
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from . import data
from .infer import BASE, update_json
from .lora import add_lora, save_adapter, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="'gold' or a jsonl of {problem, solution}")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--budget_tokens", type=int)
    ap.add_argument("--save_epochs", action="store_true")
    ap.add_argument("--replay", help="jsonl of {prompt, solution} replay pairs")
    ap.add_argument("--replay_frac", type=float, default=0.2, help="share of training tokens from replay")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_tokens", type=int, default=3000)
    ap.add_argument("--out", default="/workspace/runs/baselines")
    a = ap.parse_args()

    if a.data == "gold":
        pairs = [dict(problem=r["problem"], solution=r["solution"]) for r in data.math_train_dev()[0]]
    else:
        pairs = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    n_replay = 0
    if a.replay:
        # add replay pairs until they make up replay_frac of the tokens in one pass
        from .lora import encode
        task_tokens = sum(len(encode(tok, p)[0]) for p in pairs)
        want = a.replay_frac / (1 - a.replay_frac) * task_tokens
        pool = [json.loads(l) for l in Path(a.replay).read_text().splitlines() if l]
        random.Random(a.seed).shuffle(pool)
        got = 0
        for r in pool:
            if got >= want:
                break
            pairs.append(r)
            got += len(encode(tok, r)[0])
            n_replay += 1
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    wrapped = add_lora(model, a.rank, 2 * a.rank)
    adapters = Path(a.out) / "adapters"
    adapters.mkdir(parents=True, exist_ok=True)

    def on_epoch(e):
        if a.save_epochs:
            save_adapter(wrapped, adapters / f"{a.tag}.ep{e}.pt", a.rank, 2 * a.rank)

    t0 = time.time()
    stats = train(model, tok, wrapped, pairs, seed=a.seed, lr=a.lr, batch_tokens=a.batch_tokens,
                  epochs=a.epochs, budget_tokens=a.budget_tokens, on_epoch=on_epoch)
    save_adapter(wrapped, adapters / f"{a.tag}.pt", a.rank, 2 * a.rank)
    stats.update(seed=a.seed, rank=a.rank, alpha=2 * a.rank, data=a.data, replay_examples=n_replay,
                 total_seconds=round(time.time() - t0, 1))
    update_json(Path(a.out) / "results" / f"{a.tag}.json", train=stats)
    print(f"[{a.tag}] train: {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
