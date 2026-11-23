"""Rehearse and filter: sample k solutions per question and keep the ones the votes agree on.

  B3 (real MATH train questions):  python -m dreamlite.rehearse --tag b3_s0 --seed 0

A question is kept when the largest group of equivalent answers has more than half of the
votes but not all of them (5 to 7 of 8). Up to two solutions with the majority answer are kept
per question. Gold answers are never used for filtering; they are only used afterwards to
report how often the majority answer was right.
"""
import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import data
from .answers import equiv, extract_math, normalize, vote
from .infer import BASE, generate, load_llm, update_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--style", default="boxed", choices=["boxed", "minerva"])
    ap.add_argument("--limit", type=int, help="first N questions only (for testing)")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out", default="/workspace/runs/baselines")
    a = ap.parse_args()

    out = Path(a.out) / "b3"
    out.mkdir(parents=True, exist_ok=True)
    train, _ = data.math_train_dev()
    prefix = data.math_prefix(train, a.style)
    rows = train[:a.limit] if a.limit else train
    llm = load_llm(a.model)
    t0 = time.time()
    texts, st = generate(llm, [data.math_prompt(prefix, r["problem"]) for r in rows], n=a.k,
                         temperature=0.8, max_tokens=a.max_tokens, stop=data.MATH_STOP, seed=a.seed)

    t1 = time.time()
    lo, hi = a.k // 2 + 1, a.k - 1
    hist, right, kept, recs = Counter(), defaultdict(int), [], []
    for r, ts in zip(rows, texts):
        maj, idx = vote([normalize(extract_math(t)) for t in ts])
        ok = equiv(maj, normalize(r["answer"]))
        hist[len(idx)] += 1
        right[len(idx)] += ok
        keep = lo <= len(idx) <= hi
        recs.append(dict(problem=r["problem"], majority=maj, agree=len(idx), kept=keep,
                         majority_correct=ok, level=r["level"], subject=r["type"]))
        if keep:
            kept += [dict(problem=r["problem"], solution=ts[i].strip()) for i in idx[:2]]
    (out / f"{a.tag}.all.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
    (out / f"{a.tag}.kept.jsonl").write_text("\n".join(json.dumps(x) for x in kept) + "\n")
    n_kept = sum(x["kept"] for x in recs)
    summary = dict(questions=len(rows), kept_questions=n_kept, kept_pairs=len(kept),
                   unanimous=hist[a.k], no_majority=sum(v for c, v in hist.items() if c < lo),
                   kept_label_accuracy=round(100 * sum(x["majority_correct"] for x in recs if x["kept"])
                                             / max(1, n_kept), 1),
                   by_agreement={str(c): dict(n=hist[c], majority_correct=right[c]) for c in range(a.k + 1)},
                   filter_seconds=round(time.time() - t1, 1), total_seconds=round(time.time() - t0, 1),
                   seed=a.seed, k=a.k, **st)
    update_json(Path(a.out) / "results" / f"{a.tag}.json", rehearse=summary)
    print(f"[{a.tag}] rehearse: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
