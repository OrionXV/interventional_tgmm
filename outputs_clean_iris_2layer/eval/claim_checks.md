# Clean-claim checks

- Train seed: 0; eval seeds: [101, 102, 103]; disjoint: True

## residual

- No shift: TGMM PE = 2.4082; best baseline PE = 3.1207 (kmeans); TGMM wins on PE = True
- No shift: TGMM Acc = 0.7669; best baseline Acc = 0.6611 (diag_gmm); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0165 | -0.0703 |
| mechanism@0.65 | 0.0990 | 0.0913 |
| noise@1.5 | 2.9943 | 0.0641 |
| sample_size@128 | -0.0027 | -0.0037 |

## gaussian_diag

- No shift: TGMM PE = 2.4296; best baseline PE = 4.0600 (diag_gmm); TGMM wins on PE = True
- No shift: TGMM Acc = 0.6162; best baseline Acc = 0.5806 (diag_gmm); TGMM wins on Acc = True

| Intervention | TGMM PE gap | TGMM Acc gap |
|---|---:|---:|
| prior@1.25 | 0.0165 | -0.0627 |
| mechanism@0.65 | 0.0317 | 0.0870 |
| noise@1.5 | 3.0021 | 0.0823 |
| sample_size@128 | -0.0064 | -0.0016 |

