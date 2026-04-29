# Clean-claim checks

- Train seed: 0; eval seeds: [101, 102]; disjoint: True

## residual

- No shift: TGMM PE = 1.1289; best baseline PE = 0.3210 (kmeans); TGMM wins on PE = False
- No shift: TGMM Acc = 0.7291; best baseline Acc = 0.9480 (kmeans); TGMM wins on Acc = False

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0709 | -0.1114 |
| mechanism@0.65 | -0.5139 | 0.0942 |
| noise@1.5 | 0.2458 | 0.1146 |
| sample_size@128 | -0.0050 | 0.1315 |

## gaussian_diag

- No shift: TGMM PE = 1.1252; best baseline PE = 0.5231 (em); TGMM wins on PE = False
- No shift: TGMM Acc = 0.6134; best baseline Acc = 0.7915 (em); TGMM wins on Acc = False

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0599 | -0.0998 |
| mechanism@0.65 | -0.5124 | 0.0523 |
| noise@1.5 | 0.2499 | 0.0458 |
| sample_size@128 | -0.0035 | 0.1251 |

