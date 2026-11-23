"""MMLU accuracy on a stratified 1,000-question subset, 5-shot, scored by option log-likelihood.

  python -m dreamlite.mmlu --model Qwen/Qwen3-1.7B-Base --tag b0

The prediction is whichever of " A", " B", " C", " D" has the highest next-token logit
after "Answer:". Forgetting is the change in this number relative to the base model.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from . import data
from .infer import BASE, update_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="/workspace/runs/baselines")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side, tok.truncation_side = "left", "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    options = [tok(f" {c}")["input_ids"] for c in "ABCD"]
    assert all(len(o) == 1 for o in options), options
    options = torch.tensor([o[0] for o in options]).cuda()
    head = model.get_output_embeddings()

    items = sorted(data.mmlu_subset(a.n), key=lambda x: len(x["prompt"]))
    t0, preds = time.time(), []
    with torch.no_grad():
        for i in range(0, len(items), a.batch_size):
            batch = items[i:i + a.batch_size]
            enc = tok([b["prompt"] for b in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=3072).to("cuda")
            # logits for the last position only; all positions would not fit in GPU memory
            h = model.base_model(**enc).last_hidden_state[:, -1]
            preds += head(h)[:, options].argmax(-1).tolist()
    correct = [p == b["answer"] for p, b in zip(preds, items)]
    out = Path(a.out)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    (out / "outputs" / f"{a.tag}.mmlu.jsonl").write_text("\n".join(
        json.dumps(dict(subject=b["subject"], pred=p, gold=b["answer"])) for p, b in zip(preds, items)) + "\n")
    summary = dict(accuracy=round(100 * sum(correct) / len(correct), 2), n=len(correct),
                   model=a.model, total_seconds=round(time.time() - t0, 1))
    update_json(out / "results" / f"{a.tag}.json", mmlu=summary)
    print(f"[{a.tag}] mmlu: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
