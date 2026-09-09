# Clean-claim checks

- Train seed: 0; eval seeds: [101, 102, 103]; disjoint: True

## residual

- No shift: TGMM PE = 2.4082; best baseline PE = 3.1207 (kmeans); TGMM wins on PE = True
- No shift: TGMM Acc = 0.7661; best baseline Acc = 0.6611 (diag_gmm); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0154 | -0.0715 |
| mechanism@0.65 | 0.0945 | 0.0836 |
| noise@1.5 | 2.9933 | 0.0652 |
| sample_size@128 | -0.0028 | -0.0039 |

## gaussian_diag

- No shift: TGMM PE = 2.4302; best baseline PE = 4.0600 (diag_gmm); TGMM wins on PE = True
- No shift: TGMM Acc = 0.6151; best baseline Acc = 0.5806 (diag_gmm); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0109 | -0.0638 |
| mechanism@0.65 | 0.0290 | 0.0863 |
| noise@1.5 | 3.0027 | 0.0790 |
| sample_size@128 | -0.0061 | -0.0019 |

