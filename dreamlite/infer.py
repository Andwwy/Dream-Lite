"""Thin wrappers around vLLM offline generation, with throughput bookkeeping."""
import json
import time
from pathlib import Path

BASE = "Qwen/Qwen3-1.7B-Base"


def load_llm(model, max_num_seqs=256):
    from vllm import LLM
    return LLM(model=model, dtype="bfloat16", max_model_len=4096, seed=0,
               gpu_memory_utilization=0.90, max_num_seqs=max_num_seqs)


def generate(llm, prompts, with_finish=False, **sampling):
    """Returns (list of lists of completion texts, stats dict).

    with_finish=True returns (text, finish_reason) pairs instead of texts;
    finish_reason is "length" when the completion ran into max_tokens.
    """
    from vllm import SamplingParams
    t = time.time()
    outs = llm.generate(prompts, SamplingParams(**sampling))
    dt = time.time() - t
    comps = [c for o in outs for c in o.outputs]
    n_out = sum(len(c.token_ids) for c in comps)
    stats = dict(seconds=round(dt, 1), sequences=len(comps), output_tokens=n_out,
                 output_tokens_per_s=round(n_out / dt), mean_output_tokens=round(n_out / len(comps), 1),
                 hit_max_tokens=round(sum(c.finish_reason == "length" for c in comps) / len(comps), 3))
    if with_finish:
        return [[(c.text, c.finish_reason) for c in o.outputs] for o in outs], stats
    return [[c.text for c in o.outputs] for o in outs], stats


def update_json(path, **kw):
    """Merge keys into a JSON file (created if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(path.read_text()) if path.exists() else {}
    d.update(kw)
    path.write_text(json.dumps(d, indent=2))
    return d
