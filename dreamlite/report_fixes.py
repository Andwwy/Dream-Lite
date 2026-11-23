"""Results table for the follow-up runs (dev-stopped training and replay), as Markdown.

  python -m dreamlite.report_fixes --base results/baselines --dream results/dream --out results/fixes
"""
import argparse
import json
import statistics
from pathlib import Path

from .report import correct, diff_ci

SETS = ["math500", "gsm8k", "mmlu"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/workspace/runs/baselines")
    ap.add_argument("--dream", default="/workspace/runs/dream")
    ap.add_argument("--out", default="/workspace/runs/fixes")
    a = ap.parse_args()
    base, dream, out = Path(a.base), Path(a.dream), Path(a.out)
    seeds = lambda t: [f"{t}_s{s}" for s in range(3)]
    methods = [
        ("B2 supervised LoRA, full budget", base, seeds("b2")),
        ("B2 + 20% replay", out, seeds("b2r")),
        ("B3 self-improve, full budget", base, seeds("b3")),
        ("B3, dev-stopped", out, seeds("b3ds")),
        ("B3 + 20% replay", out, seeds("b3r")),
        ("Dreaming, full budget", dream, ["dream_s0_n1"]),
        ("Dreaming, dev-stopped", out, ["dream_devstop"]),
        ("Dreaming + 20% replay", out, ["dreamr_s0"]),
    ]
    def res(d, t):  # on the pod the json files sit in results/, in the repo copy next to outputs/
        f = d / "results" / f"{t}.json"
        return json.loads((f if f.exists() else d / f"{t}.json").read_text())
    b0 = {s: correct(base, "b0", s) for s in SETS}

    print("| Method | Seeds | MATH-500 | vs B0 (95% CI) | GSM8K | vs B0 (95% CI) | MMLU | vs B0 (95% CI) |")
    print("|---|---|---|---|---|---|---|---|")
    print("| B0 base, greedy | 1 | " + " | | ".join(f"{res(base, 'b0')[s]['accuracy']:.1f}" for s in SETS) + " | |")
    for label, d, tags in methods:
        cells = [label, str(len(tags))]
        for s in SETS:
            accs = [res(d, t)[s]["accuracy"] for t in tags]
            per = [correct(d, t, s) for t in tags]
            lo, hi = diff_ci([sum(c) / len(c) for c in zip(*per)], b0[s])
            cells += [f"{statistics.mean(accs):.1f}" + (f" ± {statistics.stdev(accs):.1f}" if len(accs) > 1 else ""),
                      f"{statistics.mean(accs) - res(base, 'b0')[s]['accuracy']:+.1f} [{lo:+.1f}, {hi:+.1f}]"]
        print("| " + " | ".join(cells) + " |")

    print("\nB0 on the full MATH test set (5,000 problems): "
          + ", ".join(f"{t.split('_')[-1]} prompt {res(out, t)['mathtest']['accuracy']:.1f}"
                      for t in ("b0_full_boxed", "b0_full_minerva")) + ". Reported: 43.5.")
    for name in ("dream", "b3"):
        st = json.loads((out / f"{name}_stop.json").read_text())
        print(f"\n{name} (seed 0) dev accuracy by pass: "
              + ", ".join(f"{k}: {v}" for k, v in st["dev_by_pass"].items())
              + f"; full budget: {st['dev_final']}; chosen: {st['best']}. B0 dev: {res(base, 'b0')['dev']['accuracy']}.")
    for t in ("dreamr_s0", "b3r_s0", "b2r_s0"):
        tr = res(out, t)["train"]
        print(f"\n{t}: {tr['replay_examples']} replay pairs of {tr['examples']} examples, {tr['epochs_done']} passes.")


if __name__ == "__main__":
    main()
