# One dream night (seed 0), Qwen3-1.7B-Base, RTX 4090

Same LoRA settings and training-token budget as B2 and B3 (2,114,887 tokens). No replay.
Intervals are 95% paired bootstrap intervals of the difference from B0 on the same problems.

| Method | MATH-500 | vs B0 | GSM8K | vs B0 | MMLU | vs B0 |
|---|---|---|---|---|---|---|
| B0 base, greedy | 35.4 | | 73.3 | | 64.0 | |
| Dreaming, 1 night | 34.0 | -1.4 [-5.6, +3.0] | 64.7 | -8.6 [-11.2, -6.1] | 62.0 | -2.0 [-3.9, -0.1] |

| Step | Time | Result |
|---|---|---|
| Imagine | 3 min | 7,200 prompts -> 4,184 valid problems (2,798 cut off or too long, 210 duplicates, 3 overlapping a test problem, 5 empty); first 4,000 used |
| Rehearse + filter | 18 min | 594 of 4,000 kept (15%), 1,188 pairs; 3,287 with no majority; 119 unanimous |
| Consolidate | 3 min | 762 steps = 10.6 passes over the kept pairs; training loss 0.31 -> 0.03 |
| Merge + evaluate | 3 min | MATH-500, GSM8K, MMLU |

Whole night: 29 minutes. Problems by vote agreement (of 8): 0: 12, 1: 1159, 2: 1158, 3: 615, 4: 343, 5: 245, 6: 188, 7: 161, 8: 119.

`nights/` holds the 4,000 dreamed problems, the vote record for each, and the kept pairs.
