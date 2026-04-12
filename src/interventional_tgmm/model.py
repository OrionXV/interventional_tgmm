from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class TGMMNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.d, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.n_heads,
            dim_feedforward=4 * cfg.hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.slot_queries = nn.Parameter(torch.randn(cfg.k, cfg.hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.slot_ff = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
        )
        self.mean_head = nn.Linear(cfg.hidden_dim, cfg.d)
        self.weight_head = nn.Linear(cfg.hidden_dim, 1)
        self.scale_head = nn.Linear(cfg.hidden_dim, cfg.d) if cfg.predict_scales else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape [batch, n, d], got {tuple(x.shape)}")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask to have shape [batch, n], got {tuple(mask.shape)}")

        h = self.input_proj(x)
        key_padding_mask = ~mask
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        queries = self.slot_queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        slots, _ = self.cross_attn(
            query=queries,
            key=h,
            value=h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        slots = slots + self.slot_ff(slots)

        means = self.mean_head(slots)
        weight_logits = self.weight_head(slots).squeeze(-1)
        outputs: dict[str, torch.Tensor] = {"means": means, "weight_logits": weight_logits}
        if self.scale_head is not None:
            outputs["log_scales"] = self.scale_head(slots)
        return outputs

    def decode_scales(self, log_scales: torch.Tensor) -> torch.Tensor:
        return F.softplus(log_scales) + float(self.cfg.min_scale)

    def config_dict(self) -> dict[str, int | float | bool]:
        return asdict(self.cfg)
