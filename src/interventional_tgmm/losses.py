from __future__ import annotations

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

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
    """Permutation-invariant loss via exact linear assignment.

    The loss decomposes as a sum over component matches, so we can find the
    globally optimal permutation in O(K^3) with Hungarian matching instead of
    enumerating K! permutations.
    """
    batch_size, k, _ = pred_means.shape

    best_losses = []
    best_mean_losses = []
    best_pi_losses = []
    best_scale_losses = []
    best_perms = []
    use_scale_loss = pred_log_scales is not None and true_scales is not None

    for batch_idx in range(batch_size):
        pred_means_b = pred_means[batch_idx]
        true_means_b = true_means[batch_idx]
        pred_weight_logits_b = pred_weight_logits[batch_idx]
        true_weights_b = true_weights[batch_idx]

        # Pairwise MSE across dimensions, scaled so summing assigned pairs
        # matches the "reduction=mean" behavior over all K*D entries.
        mean_pair_cost = ((pred_means_b[:, None, :] - true_means_b[None, :, :]) ** 2).mean(dim=-1) / float(k)

        # Pairwise contribution to weight cross-entropy term.
        pred_log_probs = F.log_softmax(pred_weight_logits_b, dim=-1)
        pi_pair_cost = -true_weights_b[None, :] * pred_log_probs[:, None]

        if use_scale_loss:
            assert pred_log_scales is not None and true_scales is not None
            pred_scales_b = F.softplus(pred_log_scales[batch_idx]) + float(min_scale)
            true_scales_b = true_scales[batch_idx]
            scale_pair_cost = ((pred_scales_b[:, None, :] - true_scales_b[None, :, :]) ** 2).mean(dim=-1) / float(k)
        else:
            scale_pair_cost = torch.zeros_like(mean_pair_cost)

        total_pair_cost = (
            mean_loss_weight * mean_pair_cost
            + pi_loss_weight * pi_pair_cost
            + scale_loss_weight * scale_pair_cost
        )
        row_ind, col_ind = linear_sum_assignment(total_pair_cost.detach().cpu().numpy())

        # Permutation format matches earlier code: perm[j] gives the predicted
        # component index aligned to true component j.
        perm = torch.empty(k, device=pred_means.device, dtype=torch.long)
        perm[torch.from_numpy(col_ind).to(device=pred_means.device, dtype=torch.long)] = torch.from_numpy(
            row_ind
        ).to(device=pred_means.device, dtype=torch.long)

        pm = pred_means_b[perm]
        pw = pred_weight_logits_b[perm]
        mean_loss = F.mse_loss(pm, true_means_b, reduction="mean")
        pi_loss = -(true_weights_b * F.log_softmax(pw, dim=-1)).sum()
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

        best_losses.append(total_loss)
        best_mean_losses.append(mean_loss)
        best_pi_losses.append(pi_loss)
        best_scale_losses.append(scale_loss)
        best_perms.append(perm)

    return {
        "loss": torch.stack(best_losses).mean(),
        "mean_loss": torch.stack(best_mean_losses).mean(),
        "pi_loss": torch.stack(best_pi_losses).mean(),
        "scale_loss": torch.stack(best_scale_losses).mean(),
        "permutation": torch.stack(best_perms),
    }
