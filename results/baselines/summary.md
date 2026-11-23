| Method | Seeds | MATH-500 | vs B0 (95% CI) | GSM8K | vs B0 (95% CI) | MMLU | vs B0 (95% CI) |
|---|---|---|---|---|---|---|---|
| B0 base, greedy | 1 | 35.4 |  | 73.3 |  | 64.0 |  |
| B1 majority vote, k=8 | 1 | 43.4 | +8.0 [+4.0, +12.2] | 79.5 | +6.2 [+4.1, +8.3] | same model as B0 |  |
| B2 supervised LoRA | 3 | 35.9 ± 1.2 | +0.5 [-2.9, +3.9] | 69.7 ± 1.4 | -3.6 [-5.8, -1.3] | 64.1 ± 0.3 | +0.1 [-1.1, +1.3] |
| B3 self-improve, real questions | 3 | 36.4 ± 0.7 | +1.0 [-2.5, +4.6] | 73.1 ± 0.7 | -0.2 [-2.1, +1.7] | 63.4 ± 0.9 | -0.6 [-1.9, +0.8] |
| B4 Qwen3-4B-Base, greedy | 1 | 50.0 | +14.6 [+10.4, +19.0] | 87.1 | +13.8 [+11.6, +16.1] | not run |  |

Training budget: 1 epoch of gold solutions = 2,114,887 tokens. B2 dev accuracy after 1/2/3 epochs: {'1': 43.8, '2': 41.8, '3': 37.4}. B0 dev accuracy: 45.2.

| B3 seed | Questions kept | Kept pairs | Kept labels correct | No majority | Unanimous | Epochs over kept pairs |
|---|---|---|---|---|---|---|
| 0 | 1467 of 7000 | 2934 | 95.4% | 5146 | 387 | 4.52 |
| 1 | 1444 of 7000 | 2888 | 94.8% | 5188 | 368 | 4.59 |
| 2 | 1487 of 7000 | 2974 | 94.8% | 5158 | 355 | 4.43 |

B3 seed 0, majority-answer accuracy by vote agreement (of 8): 0: 0% (n=2), 1: 4% (n=1688), 2: 25% (n=1776), 3: 51% (n=1028), 4: 71% (n=652), 5: 92% (n=532), 6: 96% (n=478), 7: 99% (n=457), 8: 100% (n=387)
