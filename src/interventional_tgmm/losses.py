from __future__ import annotations

import torch
import torch.nn.functional as F

from .matching import all_permutations


def permutation_invariant_loss(
    pred_means: torch.Tensor,
    pred_weight_logits: torch.Tensor,
    true_means: torch.Tensor,
    true_weights: torch.Tensor,
    pred_log_scales: torch.Tensor | None = None,
    true_scales: torch.Tensor | None = None,
    mean_loss_weight: float = 1.0,
    pi_loss_weight: float = 1.0,
    scale_loss_weight: float = 1.0,
    min_scale: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Permutation-invariant loss for small fixed-K mixtures.

    This enumerates all K! permutations, which is perfectly fine for K=3.
    """
    batch_size, k, _ = pred_means.shape
    perms = torch.tensor(all_permutations(k), device=pred_means.device, dtype=torch.long)

    best_losses = []
    best_mean_losses = []
    best_pi_losses = []
    best_scale_losses = []
    best_perms = []
    use_scale_loss = pred_log_scales is not None and true_scales is not None

    for batch_idx in range(batch_size):
        loss_candidates = []
        mean_candidates = []
        pi_candidates = []
        scale_candidates = []
        for perm in perms:
            pm = pred_means[batch_idx, perm]
            pw = pred_weight_logits[batch_idx, perm]
            mean_loss = F.mse_loss(pm, true_means[batch_idx], reduction="mean")
            pi_loss = -(true_weights[batch_idx] * F.log_softmax(pw, dim=-1)).sum()
            if use_scale_loss:
                assert pred_log_scales is not None and true_scales is not None
                ps = F.softplus(pred_log_scales[batch_idx, perm]) + float(min_scale)
                scale_loss = F.mse_loss(ps, true_scales[batch_idx], reduction="mean")
            else:
                scale_loss = torch.zeros((), device=pred_means.device, dtype=pred_means.dtype)
            total_loss = (
                mean_loss_weight * mean_loss
                + pi_loss_weight * pi_loss
                + scale_loss_weight * scale_loss
            )
            loss_candidates.append(total_loss)
            mean_candidates.append(mean_loss)
            pi_candidates.append(pi_loss)
            scale_candidates.append(scale_loss)

        losses_tensor = torch.stack(loss_candidates)
        mean_tensor = torch.stack(mean_candidates)
        pi_tensor = torch.stack(pi_candidates)
        scale_tensor = torch.stack(scale_candidates)
        best_idx = torch.argmin(losses_tensor)

        best_losses.append(losses_tensor[best_idx])
        best_mean_losses.append(mean_tensor[best_idx])
        best_pi_losses.append(pi_tensor[best_idx])
        best_scale_losses.append(scale_tensor[best_idx])
        best_perms.append(perms[best_idx])

    return {
        "loss": torch.stack(best_losses).mean(),
        "mean_loss": torch.stack(best_mean_losses).mean(),
        "pi_loss": torch.stack(best_pi_losses).mean(),
        "scale_loss": torch.stack(best_scale_losses).mean(),
        "permutation": torch.stack(best_perms),
    }
