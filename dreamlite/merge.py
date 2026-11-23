"""Merge a LoRA adapter into the base weights and save a plain model for vLLM.

  python -m dreamlite.merge --adapter runs/adapters/b2_s0.pt --out /root/merged/b2_s0
"""
import argparse

from .infer import BASE
from .lora import merge

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merge(BASE, a.adapter, a.out)
    print(f"merged {a.adapter} -> {a.out}")
