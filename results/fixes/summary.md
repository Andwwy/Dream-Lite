| Method | Seeds | MATH-500 | vs B0 (95% CI) | GSM8K | vs B0 (95% CI) | MMLU | vs B0 (95% CI) |
|---|---|---|---|---|---|---|---|
| B0 base, greedy | 1 | 35.4 | | 73.3 | | 64.0 | |
| B2 supervised LoRA, full budget | 3 | 35.9 ± 1.2 | +0.5 [-2.9, +3.9] | 69.7 ± 1.4 | -3.6 [-5.8, -1.3] | 64.1 ± 0.3 | +0.1 [-1.1, +1.3] |
| B2 + 20% replay | 3 | 36.2 ± 0.7 | +0.8 [-2.6, +4.2] | 72.0 ± 0.8 | -1.3 [-3.3, +0.6] | 64.2 ± 0.6 | +0.2 [-0.7, +1.1] |
| B3 self-improve, full budget | 3 | 36.4 ± 0.7 | +1.0 [-2.5, +4.6] | 73.1 ± 0.7 | -0.2 [-2.1, +1.7] | 63.4 ± 0.9 | -0.6 [-1.9, +0.8] |
| B3, dev-stopped | 3 | 39.1 ± 0.4 | +3.7 [+0.3, +7.1] | 73.7 ± 0.5 | +0.4 [-1.5, +2.3] | 63.4 ± 0.5 | -0.6 [-1.8, +0.7] |
| B3 + 20% replay | 3 | 38.2 ± 0.5 | +2.8 [-0.5, +6.1] | 74.5 ± 1.0 | +1.2 [-0.7, +3.0] | 62.7 ± 1.0 | -1.3 [-2.8, +0.1] |
| Dreaming, full budget | 1 | 34.0 | -1.4 [-5.6, +3.0] | 64.7 | -8.6 [-11.2, -6.1] | 62.0 | -2.0 [-3.9, -0.1] |
| Dreaming, dev-stopped | 1 | 39.0 | +3.6 [+0.0, +7.2] | 74.5 | +1.1 [-1.0, +3.3] | 63.9 | -0.1 [-1.3, +1.1] |
| Dreaming + 20% replay | 1 | 34.8 | -0.6 [-4.8, +3.6] | 66.6 | -6.7 [-9.3, -4.2] | 62.5 | -1.5 [-3.4, +0.5] |

B0 on the full MATH test set (5,000 problems): boxed prompt 38.7, minerva prompt 41.0. Reported: 43.5.

dream (seed 0) dev accuracy by pass: 1: 47.2, 2: 47.2, 3: 43.2, 4: 45.0, 5: 41.4, 6: 40.6, 7: 41.0, 8: 40.4, 9: 43.6, 10: 39.4; full budget: 39.2; chosen: ep1. B0 dev: 45.2.

b3 (seed 0) dev accuracy by pass: 1: 44.4, 2: 44.0, 3: 44.8, 4: 43.4; full budget: 44.2; chosen: ep3. B0 dev: 45.2.

dreamr_s0: 309 replay pairs of 1497 examples, 8.47 passes.

b3r_s0: 710 replay pairs of 3644 examples, 3.62 passes.

b2r_s0: 3243 replay pairs of 10243 examples, 0.8 passes.
