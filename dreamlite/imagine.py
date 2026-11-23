"""Imagine: the model writes new problems from 4 real MATH train questions.

  python -m dreamlite.imagine --tag dream_s0_n1 --seed 0 --n 4000

Each prompt shows 4 random train questions (never their answers) and the model writes a
fifth. A problem is dropped if it ran into the token cap, is too short or too long, repeats
a train problem or an earlier dreamed one, or shares a 13-word sequence with a MATH-500 or
GSM8K test problem.
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

from . import data
from .infer import BASE, generate, load_llm, update_json

HEADER = "Below are problems from a mathematics competition.\n\n"
STOP = ["\n\nProblem:", "\nProblem:", "\n\nSolution", "\nSolution"]


def key(text):
    """Text reduced to lower-case letters and digits, to spot near-duplicates."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ngrams(text, n=13):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4000, help="valid problems wanted")
    ap.add_argument("--oversample", type=float, default=1.8, help="prompts per wanted problem")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--out", default="/workspace/runs/dream")
    a = ap.parse_args()

    out = Path(a.out) / "nights"
    out.mkdir(parents=True, exist_ok=True)
    train, _ = data.math_train_dev()
    rng = random.Random(a.seed)
    prompts = []
    for _ in range(int(a.n * a.oversample)):
        seeds = rng.sample(train, 4)
        prompts.append(HEADER + "".join(f"Problem:\n{s['problem']}\n\n" for s in seeds) + "Problem:\n")
    llm = load_llm(a.model)
    t0 = time.time()
    texts, st = generate(llm, prompts, with_finish=True, temperature=1.0, top_p=0.95, max_tokens=a.max_tokens,
                         stop=STOP, seed=a.seed)

    test_grams = set()
    for r in data.math500():
        test_grams |= ngrams(r["problem"])
    for r in data.gsm8k("test"):
        test_grams |= ngrams(r["question"])
    seen = {key(r["problem"]) for r in train}
    drop = dict(empty_or_short=0, too_long_or_cut_off=0, duplicate=0, test_overlap=0)
    problems = []
    for ((text, finish),) in texts:
        p = text.strip()
        if len(p) < 20:
            drop["empty_or_short"] += 1
        elif len(p) > 1000 or finish == "length":
            drop["too_long_or_cut_off"] += 1
        elif key(p) in seen:
            drop["duplicate"] += 1
        elif ngrams(p) & test_grams:
            drop["test_overlap"] += 1
        else:
            seen.add(key(p))
            problems.append(p)
    valid = len(problems)
    problems = problems[:a.n]
    (out / f"{a.tag}.problems.jsonl").write_text("\n".join(json.dumps(dict(problem=p)) for p in problems) + "\n")
    summary = dict(prompts=len(prompts), valid=valid, used=len(problems), dropped=drop, seed=a.seed,
                   model=a.model, total_seconds=round(time.time() - t0, 1), **st)
    update_json(Path(a.out) / "results" / f"{a.tag}.json", imagine=summary)
    print(f"[{a.tag}] imagine: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
