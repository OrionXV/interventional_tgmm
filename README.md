# Interventional TGMM

This repository implements the full project workflow for **Interventional Robustness of a Transformer-Based Unsupervised GMM Solver**.

It follows the proposal scope:
- fixed **K = 3** components by default
- main benchmark at **d = 8**
- training sample sizes **N in [32, 64]**
- a **small 3-layer transformer** rather than the full TGMM scale used in the paper
- intervention families: **prior shift**, **mechanism shift**, **noise shift**, and **sample-size shift**
- baselines: **EM**, **k-means**, and a small **spectral** estimator for isotropic mixtures

## Repository layout

- `src/interventional_tgmm/config.py` — dataclass configs
- `src/interventional_tgmm/data.py` — synthetic GMM task generation and interventions
- `src/interventional_tgmm/model.py` — small TGMM-style transformer (optional scale head for anisotropic extension)
- `src/interventional_tgmm/losses.py` — permutation-invariant training loss (means, weights, optional scales)
- `src/interventional_tgmm/baselines.py` — EM, k-means, spectral baselines
- `src/interventional_tgmm/metrics.py` — parameter error, clustering accuracy, log-likelihood
- `scripts/train_tgmm.py` — training entrypoint
- `scripts/evaluate_interventions.py` — intervention sweeps, summary tables, interventional gaps, curve SVGs
- `scripts/visualize_2d.py` — 2D qualitative visualization panels (SVG + JSON)
- `scripts/smoke_test.py` — quick end-to-end sanity check

## Quickstart

```bash
cd interventional_tgmm
pip install -e .
python scripts/smoke_test.py
```

Train a small model:

```bash
python scripts/train_tgmm.py --steps 500 --batch-size 32 --device cpu
```

Train with the optional anisotropic extension (scale prediction):

```bash
python scripts/train_tgmm.py \
  --steps 500 \
  --batch-size 32 \
  --predict-scales \
  --anisotropic-prob 0.25 \
  --device cpu
```

Evaluate it:

```bash
python scripts/evaluate_interventions.py \
  --checkpoint outputs/checkpoint_last.pt \
  --num-tasks 100 \
  --seeds 0 1 2 \
  --methods tgmm em kmeans spectral
```

This writes:
- `outputs/eval/raw_results.json` (per-task records)
- `outputs/eval/summary.json` (aggregated metrics + interventional gaps)
- `outputs/eval/main_table.md` (main quantitative table)
- `outputs/eval/intervention_curves.svg` (intervention curves)

Generate 2D visualization panels:

```bash
python scripts/visualize_2d.py \
  --checkpoint outputs/checkpoint_last.pt \
  --output outputs/eval/viz_2d.svg \
  --intervention none
```

## Staged Progression

Stage A (low-code harder benchmark):

```bash
python scripts/evaluate_interventions.py \
  --checkpoint outputs/checkpoint_last.pt \
  --preset stage_a \
  --num-tasks 100 \
  --seeds 0 1 2 \
  --methods tgmm em kmeans spectral
```

This preset includes:
- `mechanism@0.5`, `mechanism@0.35`
- `sample_size@16`, `sample_size@8`
- imbalanced prior family `dirichlet@0.2`

Stage B (harder retraining benchmark):

```bash
python scripts/train_tgmm.py --preset stage_b_d4 --steps 2000 --batch-size 64
```

or

```bash
python scripts/train_tgmm.py --preset stage_b_d2 --steps 2000 --batch-size 64
```

Stage C (stronger extension, approximated with fixed-K checkpoints):

```bash
python scripts/run_stage_progression.py --stage c --steps 2000 --num-tasks 100
```

Run all stages in sequence:

```bash
python scripts/run_stage_progression.py --stage all --steps 2000 --num-tasks 100
```

## Notes

1. The main benchmark remains isotropic (`d=8`, `K=3`, `N in [32,64]`) and includes prior/mechanism/noise/sample-size interventions.
2. The optional anisotropic extension is enabled through model scale prediction and anisotropic noise interventions.
3. The spectral baseline is isotropic-only and may return `NaN` for unsupported or ill-conditioned tasks.
