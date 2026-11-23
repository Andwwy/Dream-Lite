"""Accuracy on MATH dev, MATH-500 and GSM8K with vLLM.

  python -m dreamlite.evaluate --model Qwen/Qwen3-1.7B-Base --tag b0 --sets math500,gsm8k
  python -m dreamlite.evaluate --model Qwen/Qwen3-1.7B-Base --tag b1 --sets math500,gsm8k --k 8

k=1 is greedy decoding. k>1 samples k solutions at temperature 0.8 and scores the majority answer.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from . import data
from .answers import equiv, extract_math, gsm_number, normalize, vote
from .infer import BASE, generate, load_llm, update_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="math500,gsm8k")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--style", default="boxed", choices=["boxed", "minerva"])
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--out", default="/workspace/runs/baselines")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    train, dev = data.math_train_dev()
    prefix = data.math_prefix(train, a.style)
    sampling = dict(temperature=0.0) if a.k == 1 else dict(temperature=0.8, n=a.k, seed=0)
    llm = load_llm(a.model)

    for name in a.sets.split(","):
        t0 = time.time()
        if name in ("dev", "math500"):
            rows = dev if name == "dev" else data.math500()
            stop = data.MATH_STOP + (["I hope it is correct"] if a.style == "minerva" else [])
            texts, st = generate(llm, [data.math_prompt(prefix, r["problem"]) for r in rows],
                                 max_tokens=a.max_tokens, stop=stop, **sampling)
            recs = []
            for r, ts in zip(rows, texts):
                answers = [normalize(extract_math(t)) for t in ts]
                pred = answers[0] if a.k == 1 else vote(answers)[0]
                gold = normalize(r["answer"])
                recs.append(dict(pred=pred, gold=gold, correct=equiv(pred, gold), answers=answers,
                                 level=r.get("level"), subject=r.get("subject") or r.get("type"),
                                 text=ts[0]))
        elif name == "gsm8k":
            rows = data.gsm8k("test")
            gp = data.gsm_prefix()
            texts, st = generate(llm, [data.gsm_prompt(gp, r["question"]) for r in rows],
                                 max_tokens=512, stop=data.GSM_STOP, **sampling)
            recs = []
            for r, ts in zip(rows, texts):
                answers = [gsm_number(t) for t in ts]
                seen = Counter(x for x in answers if x is not None)
                pred = seen.most_common(1)[0][0] if seen else None
                recs.append(dict(pred=pred, gold=r["target"], answers=answers, text=ts[0],
                                 correct=pred is not None and abs(pred - r["target"]) < 1e-4))
        else:
            raise SystemExit(f"unknown set {name}")
        (out / "outputs" / f"{a.tag}.{name}.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
        acc = 100 * sum(x["correct"] for x in recs) / len(recs)
        summary = dict(accuracy=round(acc, 2), n=len(recs), no_answer=sum(x["pred"] is None for x in recs),
                       k=a.k, style=a.style, model=a.model, total_seconds=round(time.time() - t0, 1), **st)
        update_json(out / "results" / f"{a.tag}.json", **{name: summary})
        print(f"[{a.tag}] {name}: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
