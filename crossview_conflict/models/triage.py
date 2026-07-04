from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from crossview_conflict.models.backbones import build_backbone


class _EmbeddingProjector(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class CrossViewTriageNet(nn.Module):
    def __init__(
        self,
        street_backbone: str = "resnet18",
        overhead_backbone: str = "resnet18",
        embedding_dim: int = 256,
        pretrained: bool = True,
        mode: str = "crossview",
        use_generated: bool = False,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.use_generated = use_generated
        self.num_classes = int(num_classes)
        if self.num_classes < 1:
            raise ValueError("num_classes must be positive.")

        street_dim = None
        overhead_dim = None

        if mode in {"crossview", "concat", "street_only"}:
            self.street_encoder, street_dim = build_backbone(street_backbone, pretrained=pretrained)
            self.street_projector = _EmbeddingProjector(street_dim, embedding_dim)
        else:
            self.street_encoder = None
            self.street_projector = None

        if mode in {"crossview", "concat", "remote_only"}:
            self.overhead_encoder, overhead_dim = build_backbone(overhead_backbone, pretrained=pretrained)
            self.overhead_projector = _EmbeddingProjector(overhead_dim, embedding_dim)
        else:
            self.overhead_encoder = None
            self.overhead_projector = None

        fusion_dim = 0
        if mode == "street_only":
            fusion_dim = embedding_dim
        elif mode == "remote_only":
            fusion_dim = embedding_dim
        elif mode == "concat":
            fusion_dim = embedding_dim * 2
        else:
            fusion_dim = embedding_dim * 4

        if use_generated:
            fusion_dim += embedding_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, self.num_classes),
        )

    def _encode_street(self, street: torch.Tensor) -> torch.Tensor:
        if self.street_encoder is None or self.street_projector is None:
            raise RuntimeError("Street encoder is not available for this triage model.")
        return self.street_projector(self.street_encoder(street))

    def _encode_overhead(self, overhead: torch.Tensor) -> torch.Tensor:
        if self.overhead_encoder is None or self.overhead_projector is None:
            raise RuntimeError("Overhead encoder is not available for this triage model.")
        return self.overhead_projector(self.overhead_encoder(overhead))

    def forward(
        self,
        street: Optional[torch.Tensor] = None,
        overhead: Optional[torch.Tensor] = None,
        generated: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        street_embedding = None
        overhead_embedding = None

        if self.mode in {"crossview", "concat", "street_only"}:
            if street is None:
                raise ValueError("Street tensor is required for the selected triage mode.")
            street_embedding = self._encode_street(street)
        if self.mode in {"crossview", "concat", "remote_only"}:
            if overhead is None:
                raise ValueError("Overhead tensor is required for the selected triage mode.")
            overhead_embedding = self._encode_overhead(overhead)

        if self.mode == "street_only":
            features = [street_embedding]
        elif self.mode == "remote_only":
            features = [overhead_embedding]
        elif self.mode == "concat":
            assert street_embedding is not None and overhead_embedding is not None
            features = [street_embedding, overhead_embedding]
        else:
            assert street_embedding is not None and overhead_embedding is not None
            features = [
                street_embedding,
                overhead_embedding,
                torch.abs(street_embedding - overhead_embedding),
                street_embedding * overhead_embedding,
            ]

        if self.use_generated:
            if generated is None:
                raise ValueError("Generated tensor is required when use_generated=True.")
            generated_embedding = self._encode_street(generated)
            features.append(generated_embedding)
            if street_embedding is not None:
                features.append(torch.abs(street_embedding - generated_embedding))
            else:
                features.append(generated_embedding)
            if overhead_embedding is not None:
                features.append(torch.abs(overhead_embedding - generated_embedding))
            else:
                features.append(generated_embedding)

        fused = torch.cat(features, dim=-1)
        logits = self.classifier(fused)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)
        if return_embeddings:
            return logits, street_embedding, overhead_embedding
        return logits
