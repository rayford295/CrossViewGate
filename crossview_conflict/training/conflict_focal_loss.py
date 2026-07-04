"""
Conflict-Focal Loss: dynamically up-weight samples whose two view embeddings
disagree.

The same weighting is applied to binary BCE or multiclass CE, depending on the
shape of the logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConflictFocalLoss(nn.Module):
    """
    Classification loss re-weighted by real-time cross-view embedding
    disagreement.

    For each sample in the batch:
      conflict_weight = ((1 - cosine_similarity(street, overhead)) / 2) ^ gamma
      loss = (1 + conflict_weight) * base_classification_loss
    """

    def __init__(
        self,
        gamma: float = 0.5,
        pos_weight: float | None = None,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.class_weights = class_weights

    def _base_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.reshape(-1)
        if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1):
            pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
            bce = F.binary_cross_entropy_with_logits(
                logits.reshape(-1), labels.float(), pos_weight=pw, reduction="none"
            )
            if self.class_weights is not None and self.class_weights.numel() >= 2:
                weights = self.class_weights.to(logits.device)
                sample_weights = torch.where(labels.long() == 1, weights[1], weights[0])
                bce = bce * sample_weights
            return bce

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(logits, labels.long(), weight=weight, reduction="none")

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        street_feat: torch.Tensor,
        overhead_feat: torch.Tensor,
    ) -> torch.Tensor:
        base_loss = self._base_loss(logit, label)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                street_feat.detach(),
                overhead_feat.detach(),
                dim=-1,
                eps=1e-8,
            )
            conflict_weight = ((1.0 - cos_sim) / 2.0).pow(self.gamma)

        weighted = (1.0 + conflict_weight) * base_loss
        return weighted.mean()

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, pos_weight={self.pos_weight}"


class ConflictFocalLossWithStats(ConflictFocalLoss):
    """ConflictFocalLoss plus per-batch diagnostics."""

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        street_feat: torch.Tensor,
        overhead_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        base_loss = self._base_loss(logit, label)

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                street_feat.detach(), overhead_feat.detach(), dim=-1, eps=1e-8
            )
            conflict_weight = ((1.0 - cos_sim) / 2.0).pow(self.gamma)

        weighted = (1.0 + conflict_weight) * base_loss
        loss = weighted.mean()

        stats = {
            "mean_conflict_weight": conflict_weight.mean().item(),
            "frac_high_conflict": (conflict_weight > 0.3).float().mean().item(),
            "mean_base_loss": base_loss.mean().item(),
            "mean_weighted_loss": weighted.mean().item(),
        }
        return loss, stats
