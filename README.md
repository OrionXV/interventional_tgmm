# Interventional TGMM

## Report and public research snapshot

[**Read the technical report (PDF)**](paper/interventional_tgmm_report.pdf) · [**Interpretation notes and reproducibility details**](paper/README.md)

This repository now includes the May 2026 course-project report and recorded Wine, Iris, and Digits experiments. It is exploratory work, not a peer-reviewed paper. The public-release notes document an omitted fixed-geometry baseline, a residual-scale metadata mismatch, and the inapplicability of the spectral baseline to the Digits configuration. Read those notes before interpreting the archived comparisons.

The code and recorded results are preserved as a research snapshot; the full experiments have not been rerun or corrected for this release. The original Wine-focused workflow is documented below. Expanded report configurations are stored with the corresponding runs.

This repository implements the full project workflow for **Interventional Robustness of a Transformer-Based Unsupervised GMM Solver**.

It follows the proposal scope:
- fixed **K = 3** components by default
- UCI **Wine** dataset (`wine.csv`): **3 classes**, **13 continuous features**
- main benchmark at **d = 8** after PCA
- training sample sizes **N in [32, 64]**
- a **small 3-layer transformer** rather than the full TGMM scale used in the paper
- intervention families: **prior shift**, **mechanism shift**, **noise shift**, and **sample-size shift**
- baselines: **EM**, **k-means**, and a small **spectral** estimator for isotropic mixtures

## Repository layout

- `src/interventional_tgmm/config.py` — dataclass configs
- `src/interventional_tgmm/data.py` — Wine dataset loading, PCA projection, class-aware task sampling, and interventions
- `src/interventional_tgmm/model.py` — small TGMM-style transformer (optional scale head for anisotropic extension)
- `src/interventional_tgmm/losses.py` — permutation-invariant training loss (means, weights, optional scales)
- `src/interventional_tgmm/baselines.py` — EM, k-means, spectral baselines
- `src/interventional_tgmm/metrics.py` — parameter error, clustering accuracy, log-likelihood
- `scripts/train_tgmm.py` — training entrypoint
- `scripts/evaluate_interventions.py` — intervention sweeps, summary tables, interventional gaps, curve SVGs
- `scripts/visualize_2d.py` — 2D qualitative visualization panels (SVG + JSON)
- `scripts/smoke_test.py` — quick end-to-end sanity check

## Quickstart

Install dependencies (Conda):

```bash
conda create -n interventional_tgmm python=3.11 -y
conda activate interventional_tgmm
pip install -r requirements.txt
pip install -e .
```

Install dependencies (venv / system Python):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run smoke test:

```bash
cd interventional_tgmm
python scripts/smoke_test.py
```

Run an actual end-to-end quick test (keeps temp outputs for inspection):

```bash
python scripts/train_tgmm.py \
  --config config.yaml \
  --show-tmp \
  --steps 20 \
  --batch-size 8 \
  --device cpu \
  --output-dir tmp_quick_test

python scripts/evaluate_interventions.py \
  --config config.yaml \
  --show-tmp \
  --checkpoint tmp_quick_test/checkpoint_last.pt \
  --output-dir tmp_quick_eval \
  --num-tasks 5 \
  --seeds 0 \
  --methods tgmm em kmeans \
  --device cpu

python scripts/visualize_2d.py \
  --config config.yaml \
  --show-tmp \
  --checkpoint tmp_quick_test/checkpoint_last.pt \
  --output tmp_quick_eval/viz_2d.svg \
  --intervention none \
  --device cpu
```

Quick test artifacts will be under:
- `outputs/tmp/tmp_quick_test/`
- `outputs/tmp/tmp_quick_eval/`

## Configuration

All main scripts read defaults from `config.yaml` (root) via `--config`.
You can edit this file to control dataset/model/train/eval/visualization parameters in one place.
The `runtime` section controls temporary artifact behavior:
- `use_tqdm: true` enables progress bars for training/evaluation/stage runs.
- `show_tmp: false` routes `tmp*` paths into `outputs/tmp/...` and removes `outputs/tmp` after each run.
- set `show_tmp: true` to keep temporary outputs.
- if you chain scripts through `tmp*` paths (for example train then eval from a tmp checkpoint), use `show_tmp: true`.
You can also disable per run with `--no-use-tqdm`.

Example:

```bash
python scripts/train_tgmm.py --config config.yaml
python scripts/evaluate_interventions.py --config config.yaml
python scripts/visualize_2d.py --config config.yaml
python scripts/run_stage_progression.py --config config.yaml
```

CLI flags still override values from `config.yaml`.

Train a small model:

```bash
python scripts/train_tgmm.py \
  --dataset wine \
  --dataset-path wine.csv \
  --label-column Cultivars \
  --steps 500 \
  --batch-size 32 \
  --device cpu
```

Train with the optional anisotropic extension (scale prediction):

```bash
python scripts/train_tgmm.py \
  --dataset wine \
  --dataset-path wine.csv \
  --label-column Cultivars \
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
  --dataset wine \
  --dataset-path wine.csv \
  --label-column Cultivars \
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
  --dataset wine \
  --dataset-path wine.csv \
  --label-column Cultivars \
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

Stage B (harder retraining benchmark with lower PCA dimension):

```bash
python scripts/train_tgmm.py --preset stage_b_d4 --steps 2000 --batch-size 64
```

or

```bash
python scripts/train_tgmm.py --preset stage_b_d2 --steps 2000 --batch-size 64
```

Stage C (stronger extension with fixed `K=3` and anisotropic variants):

```bash
python scripts/run_stage_progression.py --stage c --steps 2000 --num-tasks 100
```

Run all stages in sequence:

```bash
python scripts/run_stage_progression.py --stage all --steps 2000 --num-tasks 100
```

## Notes

1. The benchmark is Wine-backed with standardized 13D features projected to `d=8` by PCA, and fixed `K=3`.
2. The optional anisotropic extension is enabled through model scale prediction and anisotropic noise interventions.
3. UCI Wine is a well-posed problem with well-behaved class structure, which aligns well with the fixed-`K=3` solver.
4. The spectral baseline is isotropic-only and may return `NaN` for unsupported or ill-conditioned tasks.
