#!/usr/bin/env bash
# Follow-up experiments after the first baselines and dream night. Safe to re-run.
#
#   1. B0 on the full MATH test set (5,000 problems) with both prompt styles, because the
#      reported 43.5 is on the full set.
#   2. Dev-stopped training: save the adapter after every pass, score each on dev, and test
#      the best one. Dreaming seed 0; B3 picks the pass count on seed 0 and uses it for seeds 0-2.
#   3. Replay: B2, B3 and the dream night retrained with 20% replay tokens at the full budget.
set -euo pipefail
export HF_HOME=/workspace/hf
PY=/root/venv/bin/python
B=/workspace/runs/baselines
D=/workspace/runs/dream
F=${F:-/workspace/runs/fixes}
M=/root/merged
BASE=Qwen/Qwen3-1.7B-Base
cd "$(dirname "$0")"
mkdir -p "$F/results" "$F/adapters" "$M"
T=$($PY -c "import json;print(json.load(open('$B/budget.json'))['tokens'])")

has() { $PY -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$F/results/$1.json" "$2" 2>/dev/null; }
step() { echo "=== $(date +%H:%M:%S) $*"; }

dev_eval() {  # dev_eval <tag> <adapter>
  has "$1" dev && return 0
  step "dev $1"; $PY -m dreamlite.merge --adapter "$2" --out "$M/$1"
  $PY -m dreamlite.evaluate --model "$M/$1" --tag "$1" --sets dev --out "$F"
  rm -rf "$M/$1"
}
full_eval() {  # full_eval <tag> <adapter>
  if has "$1" gsm8k && has "$1" mmlu; then return 0; fi
  step "test $1"; $PY -m dreamlite.merge --adapter "$2" --out "$M/$1"
  has "$1" gsm8k || $PY -m dreamlite.evaluate --model "$M/$1" --tag "$1" --sets math500,gsm8k --out "$F"
  has "$1" mmlu  || $PY -m dreamlite.mmlu --model "$M/$1" --tag "$1" --out "$F"
  rm -rf "$M/$1"
}
sweep() {  # sweep <name> <pairs.jsonl>: train with per-pass checkpoints, dev-score each, write <name>_stop.json
  local name=$1 pairs=$2
  has "${name}_ckpt" train || { step "train ${name}_ckpt"; $PY -m dreamlite.train --data "$pairs" --tag "${name}_ckpt" --seed 0 --budget_tokens "$T" --save_epochs --out "$F"; }
  for f in "$F"/adapters/"${name}"_ckpt.ep*.pt; do
    e=${f##*.ep}; e=${e%.pt}
    dev_eval "${name}_ep$e" "$f"
  done
  dev_eval "${name}_final" "$F/adapters/${name}_ckpt.pt"
  $PY - "$F" "$name" <<'EOF'
import glob, json, re, sys
from pathlib import Path
f, name = Path(sys.argv[1]), sys.argv[2]
dev = {}
for p in glob.glob(str(f / "results" / f"{name}_ep*.json")):
    dev[int(re.search(r"_ep(\d+)\.json", p).group(1))] = json.loads(Path(p).read_text())["dev"]["accuracy"]
final = json.loads((f / "results" / f"{name}_final.json").read_text())["dev"]["accuracy"]
best = max(dev, key=lambda e: (dev[e], -e))
out = dict(dev_by_pass=dict(sorted(dev.items())), dev_final=final,
           best="final" if final > dev[best] else f"ep{best}", best_pass=best)
(f / f"{name}_stop.json").write_text(json.dumps(out, indent=2))
print(f"[{name}] stop: {json.dumps(out)}")
EOF
}
best() { $PY -c "import json;print(json.load(open('$F/$1_stop.json'))['$2'])"; }
best_adapter() {  # the checkpoint with the best dev accuracy; the full-budget adapter if nothing beats it
  if [ "$(best "$1" best)" = final ]; then echo "$F/adapters/$1_ckpt.pt"; else echo "$F/adapters/$1_ckpt.$(best "$1" best).pt"; fi
}

# ---- 1. B0 on the full MATH test set
has b0_full_boxed mathtest   || { step "B0 full test, boxed prompt";   $PY -m dreamlite.evaluate --tag b0_full_boxed   --sets mathtest --style boxed   --out "$F"; }
has b0_full_minerva mathtest || { step "B0 full test, minerva prompt"; $PY -m dreamlite.evaluate --tag b0_full_minerva --sets mathtest --style minerva --out "$F"; }

# ---- 2. dev-stopped dreaming and B3
sweep dream "$D/nights/dream_s0_n1.kept.jsonl"
full_eval dream_devstop "$(best_adapter dream)"

sweep b3 "$B/b3/b3_s0.kept.jsonl"
if [ "$(best b3 best)" = final ]; then
  step "B3: the full budget is already the best dev checkpoint; dev-stopped B3 equals the existing B3 runs"
else
  K=$(best b3 best_pass)
  full_eval b3ds_s0 "$F/adapters/b3_ckpt.ep$K.pt"
  for s in 1 2; do
    has b3ds_s$s train || { step "train b3ds_s$s ($K passes)"; $PY -m dreamlite.train --data "$B/b3/b3_s$s.kept.jsonl" --tag b3ds_s$s --seed $s --epochs "$K" --out "$F"; }
    full_eval b3ds_s$s "$F/adapters/b3ds_s$s.pt"
  done
fi

# ---- 3. replay at the full budget
[ -s "$F/replay.jsonl" ] || { step "replay data"; $PY -m dreamlite.replay --out "$F/replay.jsonl"; }
has dreamr_s0 train || { step "train dreamr_s0"; $PY -m dreamlite.train --data "$D/nights/dream_s0_n1.kept.jsonl" --tag dreamr_s0 --seed 0 --budget_tokens "$T" --replay "$F/replay.jsonl" --out "$F"; }
full_eval dreamr_s0 "$F/adapters/dreamr_s0.pt"
for s in 0 1 2; do
  has b3r_s$s train || { step "train b3r_s$s"; $PY -m dreamlite.train --data "$B/b3/b3_s$s.kept.jsonl" --tag b3r_s$s --seed $s --budget_tokens "$T" --replay "$F/replay.jsonl" --out "$F"; }
  full_eval b3r_s$s "$F/adapters/b3r_s$s.pt"
  has b2r_s$s train || { step "train b2r_s$s"; $PY -m dreamlite.train --data gold --tag b2r_s$s --seed $s --budget_tokens "$T" --replay "$F/replay.jsonl" --out "$F"; }
  full_eval b2r_s$s "$F/adapters/b2r_s$s.pt"
done
step "all fixes done"
