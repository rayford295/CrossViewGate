"""
Insight 4: Conflict-Focal Loss — dynamically up-weight conflict samples.

Standard BCE weights all training samples equally. Conflict cases (3.4%
of wildfire) are nearly invisible. This loss rebalances toward them by
measuring real-time view disagreement from the embedding distance.

Unlike RetinaNet focal loss (which targets easy/hard predictions), this
targets agreement/conflict — a training distribution mismatch specific to
cross-view learning.

Usage in train_triage.py:
    from crossview_conflict.training.conflict_focal_loss import ConflictFocalLoss
    criterion = ConflictFocalLoss(gamma=0.5)
    ...
    loss = criterion(logit, label, street_feat, overhead_feat)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConflictFocalLoss(nn.Module):
    """
    BCE loss re-weighted by real-time cross-view embedding disagreement.

    For each sample in the batch:
      conflict_weight = ((1 - cosine_similarity(street, overhead)) / 2) ^ gamma
      loss = (1 + conflict_weight) * BCE(logit, label)

    gamma=0: standard BCE (no reweighting)
    gamma=0.5: mild conflict emphasis (recommended starting point)
    gamma=1.0: strong conflict emphasis

    Args:
        gamma: exponent controlling the strength of conflict weighting.
               Higher values push the model harder toward conflict cases.
        pos_weight: optional positive-class weight for class imbalance
                    (same as torch BCE pos_weight argument).
    """

    def __init__(self, gamma: float = 0.5, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.pos_weight = (
            torch.tensor([pos_weight]) if pos_weight is not None else None
        )

    def forward(
        self,
        logit: torch.Tensor,           # [B]   raw (pre-sigmoid) prediction
        label: torch.Tensor,           # [B]   binary {0, 1}
        street_feat: torch.Tensor,     # [B, D] projected street embedding
        overhead_feat: torch.Tensor,   # [B, D] projected overhead embedding
    ) -> torch.Tensor:
        pw = self.pos_weight.to(logit.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(
            logit, label.float(), pos_weight=pw, reduction="none"
        )  # [B]

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                street_feat.detach(),
                overhead_feat.detach(),
                dim=-1,
                eps=1e-8,
            )  # [B], in [-1, 1]
            # Map to [0, 1]: 0 = identical direction, 1 = opposite
            conflict_weight = ((1.0 - cos_sim) / 2.0).pow(self.gamma)  # [B]

        weighted = (1.0 + conflict_weight) * bce
        return weighted.mean()

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, pos_weight={self.pos_weight}"


class ConflictFocalLossWithStats(ConflictFocalLoss):
    """
    Same as ConflictFocalLoss but also returns per-batch diagnostics.
    Useful for logging during training to confirm the reweighting is active.

    Returns:
        loss:  scalar tensor (same as ConflictFocalLoss)
        stats: dict with mean_conflict_weight, mean_bce, frac_high_conflict
    """

    def forward(
        self,
        logit: torch.Tensor,
        label: torch.Tensor,
        street_feat: torch.Tensor,
        overhead_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        pw = self.pos_weight.to(logit.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(
            logit, label.float(), pos_weight=pw, reduction="none"
        )

        with torch.no_grad():
            cos_sim = F.cosine_similarity(
                street_feat.detach(), overhead_feat.detach(), dim=-1, eps=1e-8
            )
            conflict_weight = ((1.0 - cos_sim) / 2.0).pow(self.gamma)

        weighted = (1.0 + conflict_weight) * bce
        loss = weighted.mean()

        stats = {
            "mean_conflict_weight": conflict_weight.mean().item(),
            "frac_high_conflict": (conflict_weight > 0.3).float().mean().item(),
            "mean_bce": bce.mean().item(),
            "mean_weighted_bce": weighted.mean().item(),
        }
        return loss, stats
