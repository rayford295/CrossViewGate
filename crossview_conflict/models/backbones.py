from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _TorchHubEmbeddingBackbone(nn.Module):
    def __init__(
        self,
        repo: str,
        entrypoint: str,
        output_dim: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        try:
            self.model = self._load_model(repo=repo, entrypoint=entrypoint, pretrained=pretrained)
        except Exception as exc:  # pragma: no cover - depends on local cache / network availability
            raise RuntimeError(
                f"Unable to load backbone '{entrypoint}' from torch.hub. "
                "Make sure the DINOv2 weights are already cached or allow the first download."
            ) from exc

    @staticmethod
    def _load_model(repo: str, entrypoint: str, pretrained: bool) -> nn.Module:
        try:
            return torch.hub.load(repo, entrypoint, pretrained=pretrained)
        except TypeError:
            return torch.hub.load(repo, entrypoint)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
            for key in ("x_norm_clstoken", "cls_token", "pooler_output", "last_hidden_state"):
                if key in outputs:
                    value = outputs[key]
                    if isinstance(value, torch.Tensor) and value.ndim == 3:
                        return value[:, 0]
                    return value
            raise TypeError("Unsupported dict output from torch.hub backbone.")
        if isinstance(outputs, (tuple, list)):
            value = outputs[0]
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                return value[:, 0]
            return value
        if isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
            return outputs[:, 0]
        return outputs


class _TransformersVisionBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        try:
            from transformers import AutoConfig, CLIPVisionModel
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "Transformers is required for CLIP backbones. Install the 'transformers' package."
            ) from exc

        if pretrained:
            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = CLIPVisionModel(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0]
        raise TypeError("Unsupported output from transformers vision backbone.")


def _normalize_name(name: str) -> str:
    return name.lower().replace("/", "_").replace("-", "_")


def get_backbone_normalization(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    normalized_name = _normalize_name(name)
    if normalized_name in {"clip_vit_b_32", "clip_vit_b32"}:
        return CLIP_MEAN, CLIP_STD
    return IMAGENET_MEAN, IMAGENET_STD


def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    normalized_name = _normalize_name(name)

    if normalized_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim

    if normalized_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim

    if normalized_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return model, dim

    if normalized_name == "dinov2_vits14":
        model = _TorchHubEmbeddingBackbone(
            repo="facebookresearch/dinov2",
            entrypoint="dinov2_vits14",
            output_dim=384,
            pretrained=pretrained,
        )
        return model, model.output_dim

    if normalized_name in {"clip_vit_b_32", "clip_vit_b32"}:
        model = _TransformersVisionBackbone(
            model_name="openai/clip-vit-base-patch32",
            output_dim=768,
            pretrained=pretrained,
        )
        return model, model.output_dim

    raise KeyError(f"Unsupported backbone: {name}")
