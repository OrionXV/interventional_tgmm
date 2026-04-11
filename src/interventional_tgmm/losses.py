from __future__ import annotations

import torch
import torch.nn.functional as F

from .matching import all_permutations


def permutation_invariant_loss(
    pred_means: torch.Tensor,
    pred_weight_logits: torch.Tensor,
    true_means: torch.Tensor,
    true_weights: torch.Tensor,
    mean_loss_weight: float = 1.0,
    pi_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Permutation-invariant loss for small fixed-K mixtures.

    This enumerates all K! permutations, which is perfectly fine for K=3.
    """
    batch_size, k, _ = pred_means.shape
    perms = torch.tensor(all_permutations(k), device=pred_means.device, dtype=torch.long)

    best_losses = []
    best_mean_losses = []
    best_pi_losses = []
    best_perms = []

    for batch_idx in range(batch_size):
        loss_candidates = []
        mean_candidates = []
        pi_candidates = []
        for perm in perms:
            pm = pred_means[batch_idx, perm]
            pw = pred_weight_logits[batch_idx, perm]
            mean_loss = F.mse_loss(pm, true_means[batch_idx], reduction="mean")
            pi_loss = -(true_weights[batch_idx] * F.log_softmax(pw, dim=-1)).sum()
            total_loss = mean_loss_weight * mean_loss + pi_loss_weight * pi_loss
            loss_candidates.append(total_loss)
            mean_candidates.append(mean_loss)
            pi_candidates.append(pi_loss)

        losses_tensor = torch.stack(loss_candidates)
        mean_tensor = torch.stack(mean_candidates)
        pi_tensor = torch.stack(pi_candidates)
        best_idx = torch.argmin(losses_tensor)

        best_losses.append(losses_tensor[best_idx])
        best_mean_losses.append(mean_tensor[best_idx])
        best_pi_losses.append(pi_tensor[best_idx])
        best_perms.append(perms[best_idx])

    return {
        "loss": torch.stack(best_losses).mean(),
        "mean_loss": torch.stack(best_mean_losses).mean(),
        "pi_loss": torch.stack(best_pi_losses).mean(),
        "permutation": torch.stack(best_perms),
    }
