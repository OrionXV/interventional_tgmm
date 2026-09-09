# Clean-claim checks

- Train seed: 0; eval seeds: [101, 102, 103]; disjoint: True

## residual

- No shift: TGMM PE = 0.1625; best baseline PE = 1.9198 (kmeans); TGMM wins on PE = True
- No shift: TGMM Acc = 0.8815; best baseline Acc = 0.6562 (kmeans); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0056 | -0.0151 |
| mechanism@0.65 | 0.2413 | 0.2078 |
| noise@1.5 | 0.1501 | 0.1267 |
| sample_size@128 | -0.0003 | -0.0023 |

## gaussian_diag

- No shift: TGMM PE = 0.1638; best baseline PE = 0.8695 (kmeans); TGMM wins on PE = True
- No shift: TGMM Acc = 0.9459; best baseline Acc = 0.7624 (em); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0056 | -0.0091 |
| mechanism@0.65 | 0.2248 | 0.2065 |
| noise@1.5 | 0.1512 | 0.1173 |
| sample_size@128 | -0.0005 | -0.0020 |

