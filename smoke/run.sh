#!/usr/bin/env bash
# Runs the whole smoke test: gen -> train -> eval (base) -> eval (merged) -> report.
set -euo pipefail
export HF_HOME=/workspace/hf
PY=/root/venv/bin/python
OUT=${OUT:-/workspace/runs/smoke}
MERGED=${MERGED:-/root/smoke_merged}
cd "$(dirname "$0")"
rm -f "$OUT/timings.json"
$PY smoke.py gen --out "$OUT" "$@"
$PY smoke.py train --out "$OUT" --merged "$MERGED"
$PY smoke.py eval --out "$OUT" --model Qwen/Qwen3-1.7B-Base --tag base
$PY smoke.py eval --out "$OUT" --model "$MERGED" --tag merged
$PY smoke.py report --out "$OUT"
