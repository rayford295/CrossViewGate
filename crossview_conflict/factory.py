from __future__ import annotations

from pathlib import Path
from typing import Any

from crossview_conflict.models.triage import CrossViewTriageNet
from crossview_conflict.utils.io import load_checkpoint


def build_triage(config: dict[str, Any]) -> CrossViewTriageNet:
    return CrossViewTriageNet(
        street_backbone=config.get("street_backbone", "resnet18"),
        overhead_backbone=config.get("overhead_backbone", "resnet18"),
        embedding_dim=int(config.get("embedding_dim", 256)),
        pretrained=bool(config.get("pretrained", True)),
        mode=config.get("mode", "crossview"),
        use_generated=bool(config.get("use_generated", False)),
        num_classes=int(config.get("num_classes", 1)),
    )


def load_triage_from_checkpoint(path: str | Path, device: str = "cpu") -> tuple[CrossViewTriageNet, dict[str, Any]]:
    checkpoint = load_checkpoint(path, device=device)
    model = build_triage(checkpoint.get("config", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint
