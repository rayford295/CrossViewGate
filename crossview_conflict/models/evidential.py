"""
Innovation I2: Evidential Deep Learning fusion head.

Replaces the sigmoid binary classifier with a Dirichlet-based evidential head.
Each single-view branch produces belief masses; fusion follows Dempster-Shafer
combination rules.

Outputs:
  pred_prob     — point prediction in [0, 1]  (use for evaluation)
  alpha_fused   — combined Dirichlet parameters  (use for EDL loss)
  u_street      — per-view epistemic uncertainty in [0, 1]
  u_overhead
  u_fused       — fused uncertainty
  conflict_mass — Dempster conflict mass in [0, 1]
                  High value → Type A conflict (confident disagreement)
                  Low value + high u → Type B conflict (mutual uncertainty)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dempster-Shafer helpers
# ---------------------------------------------------------------------------

def _belief_masses(alpha: torch.Tensor) -> torch.Tensor:
    """Convert Dirichlet alpha → belief masses b = (alpha - 1) / S."""
    S = alpha.sum(-1, keepdim=True)
    return (alpha - 1) / S.clamp(min=1e-6)


def dempster_combine(alpha_a: torch.Tensor, alpha_b: torch.Tensor) -> torch.Tensor:
    """
    Dempster-Shafer combination for two Dirichlet sources.
    alpha: [B, K]  (K=2 for binary classification)
    Returns combined alpha: [B, K]
    """
    b_a = _belief_masses(alpha_a)   # [B, 2]
    b_b = _belief_masses(alpha_b)

    # Conflict mass: evidence that goes to the wrong class from both sources
    conflict = b_a[:, 0] * b_b[:, 1] + b_a[:, 1] * b_b[:, 0]  # [B]
    denom = (1.0 - conflict).clamp(min=1e-6)                     # [B]

    # Combined belief masses (normalised)
    b_fused = torch.stack([
        (b_a[:, 0] * b_b[:, 0] +
         b_a[:, 0] * (1 - b_b.sum(-1)) +
         b_b[:, 0] * (1 - b_a.sum(-1))) / denom,   # class 0
        (b_a[:, 1] * b_b[:, 1] +
         b_a[:, 1] * (1 - b_b.sum(-1)) +
         b_b[:, 1] * (1 - b_a.sum(-1))) / denom,   # class 1
    ], dim=-1)   # [B, 2]

    # Back to alpha: rescale to match combined strength
    S_combined = (alpha_a.sum(-1) + alpha_b.sum(-1)) / 2.0   # [B]
    alpha_fused = b_fused * S_combined.unsqueeze(-1) + 1.0
    return alpha_fused


# ---------------------------------------------------------------------------
# EDL loss
# ---------------------------------------------------------------------------

def edl_loss(
    alpha: torch.Tensor,
    y: torch.Tensor,
    epoch: int = 0,
    total_epochs: int = 10,
    kl_weight: float = 0.1,
) -> torch.Tensor:
    """
    Evidential NLL + annealed KL regularisation.
    alpha: [B, K]   y: [B] integer labels in {0, ..., K-1}
    """
    S = alpha.sum(-1)                                         # [B]
    y_oh = F.one_hot(y.long(), num_classes=alpha.shape[-1]).float()   # [B, K]

    # Expected cross-entropy under Dirichlet
    nll = (y_oh * (torch.digamma(S.unsqueeze(-1)) - torch.digamma(alpha))).sum(-1)

    # KL from modified alpha (remove correct-class evidence) to uniform
    annealing = min(1.0, epoch / max(1, total_epochs * 0.5))
    alpha_hat = y_oh + (1.0 - y_oh) * alpha   # preserve wrong-class evidence
    kl = _kl_dirichlet_uniform(alpha_hat)

    return (nll + annealing * kl_weight * kl).mean()


def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1,...,1) )."""
    K = alpha.shape[-1]
    S = alpha.sum(-1)
    kl = (
        torch.lgamma(S)
        - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.lgamma(alpha).sum(-1)
        + (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S.unsqueeze(-1))).sum(-1)
    )
    return kl


# ---------------------------------------------------------------------------
# Evidential fusion head
# ---------------------------------------------------------------------------

class EvidentialFusionHead(nn.Module):
    """
    Drop-in replacement for the final classifier in CrossViewTriageNet.

    Usage:
        head = EvidentialFusionHead(embedding_dim=256)
        out  = head(street_feat, overhead_feat)

        # Training:
        loss = edl_loss(out["alpha_fused"], labels, epoch=epoch, total_epochs=epochs)

        # Inference:
        pred = (out["pred_prob"] > 0.5).long()
    """

    def __init__(self, embedding_dim: int = 256, n_classes: int = 2) -> None:
        super().__init__()
        self.street_head   = nn.Linear(embedding_dim, n_classes)
        self.overhead_head = nn.Linear(embedding_dim, n_classes)

    def forward(
        self,
        street_feat: torch.Tensor,
        overhead_feat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        alpha_s = F.softplus(self.street_head(street_feat))   + 1.0  # [B, 2]
        alpha_o = F.softplus(self.overhead_head(overhead_feat)) + 1.0

        alpha_fused = dempster_combine(alpha_s, alpha_o)   # [B, 2]

        # Uncertainty: u = K / S  (high = uncertain)
        u_s = 2.0 / alpha_s.sum(-1)
        u_o = 2.0 / alpha_o.sum(-1)
        u_f = 2.0 / alpha_fused.sum(-1)

        # Conflict mass (Type A indicator)
        b_s = _belief_masses(alpha_s)
        b_o = _belief_masses(alpha_o)
        conflict_mass = b_s[:, 0] * b_o[:, 1] + b_s[:, 1] * b_o[:, 0]

        # Point prediction (expected probability of class 1)
        S_f = alpha_fused.sum(-1)
        pred_prob = (alpha_fused[:, 1] - 1.0) / (S_f - 2.0).clamp(min=1e-6)

        return {
            "pred_prob":     pred_prob.clamp(0.0, 1.0),
            "alpha_street":  alpha_s,
            "alpha_overhead": alpha_o,
            "alpha_fused":   alpha_fused,
            "u_street":      u_s,
            "u_overhead":    u_o,
            "u_fused":       u_f,
            "conflict_mass": conflict_mass,
        }
