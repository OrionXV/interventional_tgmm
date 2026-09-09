import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from interventional_tgmm.clean_benchmark import SplitTaskConfig, load_split_benchmark, sample_split_task
from interventional_tgmm.config import InterventionConfig
from interventional_tgmm.metrics import evaluate_task
from interventional_tgmm.types import GMMEstimate

for dataset, folder in [('wine', 'outputs_clean_3layer'), ('iris', 'outputs_clean_iris_3layer'), ('digits', 'outputs_clean_digits')]:
    saved = json.loads((root / folder / 'eval/summary.json').read_text())
    cfg = SplitTaskConfig(**saved['clean_task_config'])
    cfg.dataset_path = str(root / 'wine.csv') if dataset == 'wine' else cfg.dataset_path
    benchmark = load_split_benchmark(dataset=dataset, dataset_path=cfg.dataset_path, label_column=cfg.label_column, d=cfg.d, k=cfg.k, split_seed=cfg.split_seed, train_fraction=cfg.train_fraction, fit_preprocessor_on_train=True)
    print('DATASET', dataset)
    for generator in ['residual', 'gaussian_diag']:
        cfg.generator = generator
        metrics = []
        for seed in [101, 102, 103]:
            rng = np.random.default_rng(seed)
            for _ in range(100):
                task = sample_split_task(cfg, benchmark, 'test', rng, InterventionConfig(kind='none'))
                estimate = GMMEstimate(weights=benchmark.train.weights.copy(), means=benchmark.train.means.copy(), assumed_sigma=task.params.sigma)
                metrics.append(evaluate_task(task, estimate))
        ref = saved['summary'][generator]['tgmm']['none']
        print(json.dumps({'generator':generator, 'fixed_train_geometry_accuracy':float(np.mean([m['cluster_acc'] for m in metrics])), 'tgmm_reported_accuracy':ref['cluster_acc']['mean'], 'fixed_train_geometry_parameter_error':float(np.mean([m['parameter_error'] for m in metrics])), 'tgmm_reported_parameter_error':ref['parameter_error']['mean']}))
    # Exact population scales of the finite empirical-residual generator,
    # obtained from its known per-class residual pools rather than simulation.
    cfg.generator = 'residual'
    task = sample_split_task(cfg, benchmark, 'test', np.random.default_rng(101), InterventionConfig(kind='none'))
    raw = benchmark.test.scales
    normalized = raw * (cfg.sigma / raw.mean())
    actual_scales = np.stack([np.std(residuals, axis=0) * task.params.sigma_diag[k] / normalized[k] for k, residuals in enumerate(benchmark.test.residuals)])
    valid = task.params.sigma_diag > 1e-5
    print('RESIDUAL_ACTUAL_TO_RECORDED_SCALE_RATIO',float(np.median(actual_scales[valid] / task.params.sigma_diag[valid])))
