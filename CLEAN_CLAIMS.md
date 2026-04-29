# Clean-claims experiment suite

This suite is meant to test a narrower and cleaner version of the report's claims.

## What it changes

1. **Held-out geometry**
   - The Wine data are split **within each class** into train and test subsets.
   - Preprocessing is fit on the train split and applied to both splits.
   - TGMM trains on tasks sampled from the **train split** and is evaluated on tasks sampled from the **test split**.

2. **Disjoint RNG seeds**
   - Training and evaluation use different random seeds by default.
   - `scripts/check_task_overlap.py` hashes sampled tasks and checks for exact train/eval overlap.

3. **Fairer benchmark comparisons**
   - The evaluation can be run on two task generators:
     - `residual`: resample empirical class residuals from the held-out split.
     - `gaussian_diag`: sample diagonal Gaussians with the held-out class means/scales.
   - In addition to the original isotropic baselines, the suite adds a **diagonal-covariance GaussianMixture** baseline (`diag_gmm`).

## Recommended workflow

```bash
python scripts/train_clean_tgmm.py \
  --output-dir outputs_clean/train \
  --seed 0 --split-seed 11 --train-fraction 0.7 \
  --generator residual --steps 500 --batch-size 32

python scripts/check_task_overlap.py \
  --output-dir outputs_clean/overlap \
  --train-seed 0 --eval-seeds 101 102 103 \
  --split-seed 11 --train-fraction 0.7

python scripts/evaluate_clean_claims.py \
  --checkpoint outputs_clean/train/checkpoint_last.pt \
  --output-dir outputs_clean/eval \
  --eval-split test \
  --seeds 101 102 103 \
  --num-tasks 100
```

Or run the whole thing with:

```bash
python scripts/run_clean_claims_suite.py --output-root outputs_clean_suite
```

## Outputs

- `train/checkpoint_last.pt`: held-out-split TGMM checkpoint.
- `overlap/overlap_report.json`: exact-hash overlap audit.
- `eval/main_table.md`: evaluation table for each generator.
- `eval/claim_checks.md`: narrow claim checks (for example, whether TGMM still beats the best baseline under no shift).

## Intended claim language

The clean suite is designed to support a **narrower claim** like:

> On a held-out Wine-backed semisynthetic benchmark with disjoint train/eval seeds, TGMM can be compared fairly against isotropic and diagonal-covariance baselines, and any observed advantage should be read as benchmark-specific rather than as a general statement about all GMM solvers.

