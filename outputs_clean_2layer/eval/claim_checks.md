# Clean-claim checks

- Train seed: 0; eval seeds: [101, 102, 103]; disjoint: True

## residual

- No shift: TGMM PE = 0.2718; best baseline PE = 0.6015 (kmeans); TGMM wins on PE = True
- No shift: TGMM Acc = 0.9302; best baseline Acc = 0.8420 (kmeans); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0095 | -0.0019 |
| mechanism@0.65 | 0.0429 | 0.1242 |
| noise@1.5 | 0.2510 | 0.1222 |
| sample_size@128 | -0.0040 | 0.0017 |

## gaussian_diag

- No shift: TGMM PE = 0.2698; best baseline PE = 0.5897 (em); TGMM wins on PE = True
- No shift: TGMM Acc = 0.8971; best baseline Acc = 0.8133 (em); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0192 | -0.0285 |
| mechanism@0.65 | 0.0385 | 0.1306 |
| noise@1.5 | 0.2553 | 0.1219 |
| sample_size@128 | -0.0043 | 0.0029 |

