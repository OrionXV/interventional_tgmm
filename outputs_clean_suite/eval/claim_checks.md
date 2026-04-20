# Clean-claim checks

- Train seed: 0; eval seeds: [101, 102, 103]; disjoint: True

## residual

- No shift: TGMM PE = 0.2740; best baseline PE = 0.6015 (kmeans); TGMM wins on PE = True
- No shift: TGMM Acc = 0.9308; best baseline Acc = 0.8420 (kmeans); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0087 | -0.0006 |
| mechanism@0.65 | 0.0384 | 0.1255 |
| noise@1.5 | 0.2496 | 0.1215 |
| sample_size@128 | -0.0037 | 0.0024 |

## gaussian_diag

- No shift: TGMM PE = 0.2712; best baseline PE = 0.5897 (em); TGMM wins on PE = True
- No shift: TGMM Acc = 0.8970; best baseline Acc = 0.8133 (em); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0181 | -0.0292 |
| mechanism@0.65 | 0.0402 | 0.1305 |
| noise@1.5 | 0.2530 | 0.1209 |
| sample_size@128 | -0.0037 | 0.0022 |

