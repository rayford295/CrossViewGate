from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from crossview_conflict.training.conflict_focal_loss import ConflictFocalLoss
from crossview_conflict.training.metrics import classification_metrics


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1):
        bce = F.binary_cross_entropy_with_logits(logits.reshape(-1), target.float(), reduction="none")
        if class_weights is not None and class_weights.numel() >= 2:
            weights = class_weights.to(logits.device)
            sample_weights = torch.where(target.long() == 1, weights[1], weights[0])
            bce = bce * sample_weights
        return bce.mean()
    weight = class_weights.to(logits.device) if class_weights is not None else None
    return F.cross_entropy(logits, target.long(), weight=weight)


def train_triage_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    conflict_gamma: float | None = None,
    class_weights: torch.Tensor | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    criterion = (
        ConflictFocalLoss(gamma=conflict_gamma, class_weights=class_weights)
        if conflict_gamma is not None
        else None
    )
    for batch in tqdm(dataloader, desc="train-triage", leave=False):
        target = batch["target"].to(device)
        kwargs = {}
        if "street" in batch:
            kwargs["street"] = batch["street"].to(device)
        if "overhead" in batch:
            kwargs["overhead"] = batch["overhead"].to(device)
        if "generated" in batch:
            kwargs["generated"] = batch["generated"].to(device)
        if criterion is not None:
            logits, street_embedding, overhead_embedding = model(return_embeddings=True, **kwargs)
            if street_embedding is None or overhead_embedding is None:
                loss = classification_loss(logits, target, class_weights=class_weights)
            else:
                loss = criterion(logits, target, street_embedding, overhead_embedding)
        else:
            logits = model(**kwargs)
            loss = classification_loss(logits, target, class_weights=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * target.size(0)
    return running_loss / max(len(dataloader.dataset), 1)


@torch.no_grad()
def eval_triage(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    class_names: list[str] | None = None,
) -> dict[str, object]:
    model.eval()
    logits_list = []
    targets_list = []
    for batch in tqdm(dataloader, desc="eval-triage", leave=False):
        target = batch["target"].to(device)
        kwargs = {}
        if "street" in batch:
            kwargs["street"] = batch["street"].to(device)
        if "overhead" in batch:
            kwargs["overhead"] = batch["overhead"].to(device)
        if "generated" in batch:
            kwargs["generated"] = batch["generated"].to(device)
        logits = model(**kwargs)
        logits_list.append(logits.cpu())
        targets_list.append(target.cpu())
    return classification_metrics(torch.cat(logits_list), torch.cat(targets_list), class_names=class_names)
