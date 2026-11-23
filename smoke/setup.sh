#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (PyTorch template). Takes a few minutes.
# The venv lives on the container disk (fast, erased when the pod stops);
# model weights, datasets and results live on the /workspace volume.
set -euo pipefail
mkdir -p /workspace/hf /workspace/runs
uv venv /root/venv --python 3.12
uv pip install --python /root/venv/bin/python vllm datasets sympy "antlr4-python3-runtime==4.11.1"
