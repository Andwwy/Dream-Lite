"""Replay data: Dolly-15k instructions answered by the base model itself.

  python -m dreamlite.replay --out /workspace/runs/fixes/replay.jsonl

Training on the base model's own answers to general instructions is meant to hold its
general behaviour in place while the adapter learns math. No outside labels are used:
only the instructions come from Dolly, the targets are the base model's greedy answers.
"""
import argparse
import json
import random
from pathlib import Path

from .infer import BASE, generate, load_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--max_tokens", type=int, default=256)
    a = ap.parse_args()

    from datasets import load_dataset
    rows = [r for r in load_dataset("databricks/databricks-dolly-15k", split="train")
            if not r["context"].strip() and 10 < len(r["instruction"]) < 400]
    rows = random.Random(0).sample(rows, min(a.n, len(rows)))
    prompts = [f"Instruction: {r['instruction'].strip()}\nResponse:\n" for r in rows]
    llm = load_llm(BASE)
    texts, st = generate(llm, prompts, temperature=0.0, max_tokens=a.max_tokens,
                         stop=["\nInstruction:", "\n\n\n"])
    pairs = [dict(prompt=p, solution=t.strip()) for p, (t,) in zip(prompts, texts) if len(t.strip()) >= 10]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(x) for x in pairs) + "\n")
    print(f"[replay] {json.dumps(dict(instructions=len(rows), pairs=len(pairs), **st))}", flush=True)


if __name__ == "__main__":
    main()
