from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from crossview_conflict.training.metrics import binary_classification_metrics


def train_triage_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="train-triage", leave=False):
        street = batch["street"].to(device)
        overhead = batch["overhead"].to(device)
        target = batch["target"].to(device)
        kwargs = {"street": street, "overhead": overhead}
        if "generated" in batch:
            kwargs["generated"] = batch["generated"].to(device)
        logits = model(**kwargs)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * street.size(0)
    return running_loss / max(len(dataloader.dataset), 1)


@torch.no_grad()
def eval_triage(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> dict[str, float]:
    model.eval()
    logits_list = []
    targets_list = []
    for batch in tqdm(dataloader, desc="eval-triage", leave=False):
        street = batch["street"].to(device)
        overhead = batch["overhead"].to(device)
        target = batch["target"].to(device)
        kwargs = {"street": street, "overhead": overhead}
        if "generated" in batch:
            kwargs["generated"] = batch["generated"].to(device)
        logits = model(**kwargs)
        logits_list.append(logits.cpu())
        targets_list.append(target.cpu())
    return binary_classification_metrics(torch.cat(logits_list), torch.cat(targets_list))
