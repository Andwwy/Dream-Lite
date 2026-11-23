#!/usr/bin/env bash
# Runs every baseline on one GPU. Safe to re-run: finished steps are skipped.
#
#   B0  Qwen3-1.7B-Base, greedy
#   B1  B0 + majority vote over 8 samples
#   B4  Qwen3-4B-Base, greedy
#   B2  supervised LoRA on 7,000 MATH gold solutions, seeds 0-2
#   B3  LoRA on the model's own majority-vote solutions to the same 7,000 questions, seeds 0-2
#
# Fair comparison: B2 seed 0 is trained for 3 epochs and the epoch with the best dev accuracy
# sets the training-token budget. Every other trained run (B2, B3, and later dreaming) uses
# the same LoRA settings and the same number of training tokens.
set -euo pipefail
export HF_HOME=/workspace/hf
PY=/root/venv/bin/python
O=${O:-/workspace/runs/baselines}
M=/root/merged
BASE=Qwen/Qwen3-1.7B-Base
cd "$(dirname "$0")"
mkdir -p "$O/results" "$M"

has() {  # has <tag> <key>: is this result already there?
  $PY -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" \
    "$O/results/$1.json" "$2" 2>/dev/null
}
step() { echo "=== $(date +%H:%M:%S) $*"; }

evaluate() {  # evaluate <tag> <model> <sets> [extra args]
  local tag=$1 model=$2 sets=$3; shift 3
  has "$tag" "${sets##*,}" || { step "evaluate $tag ($sets)"; $PY -m dreamlite.evaluate --model "$model" --tag "$tag" --sets "$sets" --out "$O" "$@"; }
}
mmlu() { has "$1" mmlu || { step "mmlu $1"; $PY -m dreamlite.mmlu --model "$2" --tag "$1" --out "$O"; }; }

full_eval() {  # full_eval <tag> <adapter>: merge, test sets, MMLU, clean up
  local tag=$1 adapter=$2
  if has "$tag" gsm8k && has "$tag" mmlu; then return; fi
  step "merge $tag"; $PY -m dreamlite.merge --adapter "$adapter" --out "$M/$tag"
  evaluate "$tag" "$M/$tag" math500,gsm8k
  mmlu "$tag" "$M/$tag"
  rm -rf "$M/$tag"
}

# ---- no training
evaluate b0 $BASE dev,math500,gsm8k
mmlu b0 $BASE
evaluate b1 $BASE math500,gsm8k --k 8
evaluate b4 Qwen/Qwen3-4B-Base math500,gsm8k

# ---- B2 seed 0: 3 epochs, pick the training budget on dev
has b2_s0_sweep train || { step "train b2_s0_sweep"; $PY -m dreamlite.train --data gold --tag b2_s0_sweep --seed 0 --epochs 3 --save_epochs --out "$O"; }
for e in 1 2 3; do
  tag=b2_s0_ep$e
  has $tag dev || {
    step "merge $tag"; $PY -m dreamlite.merge --adapter "$O/adapters/b2_s0_sweep.ep$e.pt" --out "$M/$tag"
    evaluate $tag "$M/$tag" dev
    rm -rf "$M/$tag"
  }
done
[ -f "$O/budget.json" ] || $PY - "$O" <<'EOF'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
dev = {e: json.loads((o / "results" / f"b2_s0_ep{e}.json").read_text())["dev"]["accuracy"] for e in (1, 2, 3)}
best = max(dev, key=lambda e: (dev[e], -e))
tr = json.loads((o / "results" / "b2_s0_sweep.json").read_text())["train"]
budget = dict(epochs=best, tokens=best * tr["epoch_tokens"], dev_accuracy_by_epoch=dev)
(o / "budget.json").write_text(json.dumps(budget, indent=2))
print("budget:", budget)
EOF
E=$($PY -c "import json;print(json.load(open('$O/budget.json'))['epochs'])")
T=$($PY -c "import json;print(json.load(open('$O/budget.json'))['tokens'])")
step "budget: $E epochs of gold = $T training tokens"

# ---- B2, three seeds at the chosen budget (seed 0 reuses the sweep's checkpoint)
full_eval b2_s0 "$O/adapters/b2_s0_sweep.ep$E.pt"
for s in 1 2; do
  has b2_s$s train || { step "train b2_s$s"; $PY -m dreamlite.train --data gold --tag b2_s$s --seed $s --budget_tokens "$T" --out "$O"; }
  full_eval b2_s$s "$O/adapters/b2_s$s.pt"
done

# ---- B3, three seeds: rehearse, filter, train on the kept pairs for the same budget
for s in 0 1 2; do
  has b3_s$s rehearse || { step "rehearse b3_s$s"; $PY -m dreamlite.rehearse --tag b3_s$s --seed $s --out "$O"; }
  has b3_s$s train || { step "train b3_s$s"; $PY -m dreamlite.train --data "$O/b3/b3_s$s.kept.jsonl" --tag b3_s$s --seed $s --budget_tokens "$T" --out "$O"; }
  full_eval b3_s$s "$O/adapters/b3_s$s.pt"
done
step "all baselines done"
