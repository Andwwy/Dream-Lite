#!/usr/bin/env bash
# One dream night: imagine -> rehearse -> filter -> consolidate -> evaluate.
#   bash run_dream.sh [seed] [model the night starts from]
# Uses the same LoRA settings and training-token budget as B2 and B3 (budget.json).
# Safe to re-run: finished steps are skipped.
set -euo pipefail
export HF_HOME=/workspace/hf
PY=/root/venv/bin/python
SEED=${1:-0}
MODEL=${2:-Qwen/Qwen3-1.7B-Base}
O=${O:-/workspace/runs/dream}
B=${B:-/workspace/runs/baselines}
M=/root/merged
TAG=dream_s${SEED}_n1
cd "$(dirname "$0")"
mkdir -p "$O/results" "$M"
T=$($PY -c "import json;print(json.load(open('$B/budget.json'))['tokens'])")

has() { $PY -c "import json,sys; sys.exit(0 if sys.argv[2] in json.load(open(sys.argv[1])) else 1)" "$O/results/$TAG.json" "$1" 2>/dev/null; }
step() { echo "=== $(date +%H:%M:%S) $*"; }

has imagine  || { step imagine;  $PY -m dreamlite.imagine  --tag $TAG --seed $SEED --model "$MODEL" --out "$O"; }
has rehearse || { step rehearse; $PY -m dreamlite.rehearse --tag $TAG --seed $SEED --model "$MODEL" --out "$O" \
                    --questions "$O/nights/$TAG.problems.jsonl" --subdir nights; }
has train    || { step "consolidate ($T tokens)"; $PY -m dreamlite.train --data "$O/nights/$TAG.kept.jsonl" --tag $TAG --seed $SEED --budget_tokens "$T" --out "$O"; }
if ! { has gsm8k && has mmlu; }; then
  step merge; $PY -m dreamlite.merge --adapter "$O/adapters/$TAG.pt" --out "$M/$TAG"
  has gsm8k || { step evaluate; $PY -m dreamlite.evaluate --model "$M/$TAG" --tag $TAG --sets math500,gsm8k --out "$O"; }
  has mmlu  || { step mmlu;     $PY -m dreamlite.mmlu     --model "$M/$TAG" --tag $TAG --out "$O"; }
  rm -rf "$M/$TAG"
fi
step "night done"
