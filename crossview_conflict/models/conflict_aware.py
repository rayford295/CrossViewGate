"""
B2: Conflict-aware fusion architecture.

The core idea: when the two views disagree (low cosine similarity), shift
the fused representation toward the overhead view, which provides more
objective geometric evidence of structural damage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crossview_conflict.models.backbones import build_backbone


class _Projector(nn.Module):
    def __init__(self, input_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConflictAwareTriageNet(nn.Module):
    """
    Conflict-aware cross-view triage model.

    During fusion, the model measures cosine similarity between the street
    and overhead embeddings. When similarity is low (views disagree), it
    shifts the convex combination toward the overhead embedding, on the
    assumption that overhead imagery provides more objective structural
    evidence than a street-level photo of a facade.

    The explicit conflict signal (|street - overhead|) is appended alongside
    the conflict-weighted representation, so the classifier sees both the
    arbitrated representation and the raw disagreement magnitude.

    Args:
        backbone: ResNet variant name (e.g. "resnet18", "resnet50").
        embedding_dim: Projection dimension for both encoders.
        pretrained: Whether to initialise backbone from ImageNet weights.
        overhead_bias: If True, shift toward overhead under conflict (default).
                       If False, shift toward street (ablation).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        pretrained: bool = True,
        overhead_bias: bool = True,
    ) -> None:
        super().__init__()
        self.overhead_bias = overhead_bias

        self.street_encoder, feat_dim = build_backbone(backbone, pretrained=pretrained)
        self.overhead_encoder, _ = build_backbone(backbone, pretrained=pretrained)
        self.street_proj = _Projector(feat_dim, embedding_dim)
        self.overhead_proj = _Projector(feat_dim, embedding_dim)

        # Input: [conflict_weighted_repr, |diff|, hadamard]  → 3 × embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        street: torch.Tensor,
        overhead: torch.Tensor,
    ) -> torch.Tensor:
        s = self.street_proj(self.street_encoder(street))    # [B, D]
        o = self.overhead_proj(self.overhead_encoder(overhead))  # [B, D]

        # Agreement score in [0, 1]: 1 = identical direction, 0 = orthogonal
        cos_sim = F.cosine_similarity(s, o, dim=-1, eps=1e-8)        # [B]
        conflict_weight = ((1.0 - cos_sim) / 2.0).unsqueeze(-1)      # [B, 1]

        # Conflict-weighted combination
        # overhead_bias=True: weight → overhead when views conflict
        # overhead_bias=False: weight → street (ablation)
        if self.overhead_bias:
            arbitrated = (1.0 - conflict_weight) * s + conflict_weight * o
        else:
            arbitrated = conflict_weight * s + (1.0 - conflict_weight) * o

        features = torch.cat(
            [arbitrated, torch.abs(s - o), s * o],
            dim=-1,
        )
        return self.classifier(features).squeeze(-1)

    # ------------------------------------------------------------------
    # Diagnostic helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def conflict_weights(
        self,
        street: torch.Tensor,
        overhead: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-sample conflict weight in [0, 1] (no gradient)."""
        s = self.street_proj(self.street_encoder(street))
        o = self.overhead_proj(self.overhead_encoder(overhead))
        cos_sim = F.cosine_similarity(s, o, dim=-1, eps=1e-8)
        return (1.0 - cos_sim) / 2.0
