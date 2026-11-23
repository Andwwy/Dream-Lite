"""Results table for the baselines, as Markdown.

  python -m dreamlite.report --out /workspace/runs/baselines > summary.md

Accuracy is the mean over seeds, with the standard deviation across seeds. The interval is a
paired bootstrap (10,000 resamples of the test problems) of the difference from B0 on the same
problems, with per-problem correctness averaged over seeds.
"""
import argparse
import json
import random
import statistics
from pathlib import Path

METHODS = [
    ("B0 base, greedy", ["b0"]),
    ("B1 majority vote, k=8", ["b1"]),
    ("B2 supervised LoRA", ["b2_s0", "b2_s1", "b2_s2"]),
    ("B3 self-improve, real questions", ["b3_s0", "b3_s1", "b3_s2"]),
    ("B4 Qwen3-4B-Base, greedy", ["b4"]),
]


def correct(out, tag, name):
    f = out / "outputs" / f"{tag}.{name}.jsonl"
    if name == "mmlu":
        return [float(r["pred"] == r["gold"]) for r in map(json.loads, f.read_text().splitlines())]
    return [float(r["correct"]) for r in map(json.loads, f.read_text().splitlines())]


def diff_ci(a, b, n=10000):
    rng, m = random.Random(0), len(a)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(100 * sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/runs/baselines")
    out = Path(ap.parse_args().out)
    res = lambda t: json.loads((out / "results" / f"{t}.json").read_text())
    sets = ["math500", "gsm8k", "mmlu"]
    base = {s: correct(out, "b0", s) for s in sets}

    print("| Method | Seeds | MATH-500 | vs B0 (95% CI) | GSM8K | vs B0 (95% CI) | MMLU | vs B0 (95% CI) |")
    print("|---|---|---|---|---|---|---|---|")
    for label, tags in METHODS:
        cells = [label, str(len(tags))]
        for s in sets:
            if s == "mmlu" and tags[0] == "b1":
                cells += ["same model as B0", ""]
                continue
            if s not in res(tags[0]):
                cells += ["not run", ""]
                continue
            accs = [res(t)[s]["accuracy"] for t in tags]
            cell = f"{statistics.mean(accs):.1f}" + (f" ± {statistics.stdev(accs):.1f}" if len(accs) > 1 else "")
            if tags == ["b0"]:
                cells += [cell, ""]
                continue
            per = [correct(out, t, s) for t in tags]
            avg = [sum(c) / len(c) for c in zip(*per)]
            lo, hi = diff_ci(avg, base[s])
            cells += [cell, f"{statistics.mean(accs) - res('b0')[s]['accuracy']:+.1f} [{lo:+.1f}, {hi:+.1f}]"]
        print("| " + " | ".join(cells) + " |")

    b = json.loads((out / "budget.json").read_text())
    print(f"\nTraining budget: {b['epochs']} epoch of gold solutions = {b['tokens']:,} tokens. "
          f"B2 dev accuracy after 1/2/3 epochs: {b['dev_accuracy_by_epoch']}. B0 dev accuracy: {res('b0')['dev']['accuracy']}.")
    print("\n| B3 seed | Questions kept | Kept pairs | Kept labels correct | No majority | Unanimous | Epochs over kept pairs |")
    print("|---|---|---|---|---|---|---|")
    for t in METHODS[3][1]:
        r, tr = res(t)["rehearse"], res(t)["train"]
        print(f"| {r['seed']} | {r['kept_questions']} of {r['questions']} | {r['kept_pairs']} | "
              f"{r['kept_label_accuracy']}% | {r['no_majority']} | {r['unanimous']} | {tr['epochs_done']} |")
    r = res("b3_s0")["rehearse"]["by_agreement"]
    print("\nB3 seed 0, majority-answer accuracy by vote agreement (of 8): "
          + ", ".join(f"{c}: {100 * v['majority_correct'] / v['n']:.0f}% (n={v['n']})" for c, v in r.items() if v["n"]))


if __name__ == "__main__":
    main()
