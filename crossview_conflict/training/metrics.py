from __future__ import annotations

from typing import Iterable, Sequence

import torch


def retrieval_recalls(
    street_embeddings: torch.Tensor,
    overhead_embeddings: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
    target_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    similarity = street_embeddings @ overhead_embeddings.T
    ranking = torch.argsort(similarity, dim=1, descending=True)
    return ranking_recalls(ranking=ranking, ks=ks, target_indices=target_indices)


def ranking_recalls(
    ranking: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
    target_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    if target_indices is None:
        target_indices = torch.arange(ranking.size(0), device=ranking.device)
    target = target_indices.to(ranking.device).view(-1, 1)
    metrics: dict[str, float] = {}
    for k in ks:
        k = min(k, ranking.size(1))
        correct = (ranking[:, :k] == target).any(dim=1).float().mean().item()
        metrics[f"recall@{k}"] = correct
    return metrics


def haversine_distance_m(
    query_latlon: torch.Tensor,
    gallery_latlon: torch.Tensor,
) -> torch.Tensor:
    """
    Compute great-circle distance in meters between matched query and gallery points.
    Both tensors should be shaped [N, 2] with columns [lat, lon] in decimal degrees.
    """
    query_rad = torch.deg2rad(query_latlon)
    gallery_rad = torch.deg2rad(gallery_latlon)

    dlat = gallery_rad[:, 0] - query_rad[:, 0]
    dlon = gallery_rad[:, 1] - query_rad[:, 1]
    a = (
        torch.sin(dlat / 2).pow(2)
        + torch.cos(query_rad[:, 0]) * torch.cos(gallery_rad[:, 0]) * torch.sin(dlon / 2).pow(2)
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt((1 - a).clamp(min=1e-12)))
    earth_radius_m = 6_371_000.0
    return earth_radius_m * c


def distance_retrieval_metrics(
    query_latlon: torch.Tensor,
    gallery_latlon: torch.Tensor,
    ranking: torch.Tensor,
    thresholds_m: Sequence[float] = (25.0, 50.0, 100.0),
) -> dict[str, float]:
    top1_indices = ranking[:, 0]
    predicted_latlon = gallery_latlon[top1_indices]
    valid_mask = ~torch.isnan(query_latlon).any(dim=1) & ~torch.isnan(predicted_latlon).any(dim=1)
    metrics: dict[str, float] = {}
    if not valid_mask.any():
        metrics["top1_distance_error_m_mean"] = float("nan")
        metrics["top1_distance_error_m_median"] = float("nan")
        for threshold in thresholds_m:
            metrics[f"top1_within_{int(threshold)}m"] = float("nan")
        return metrics

    distances = haversine_distance_m(query_latlon[valid_mask], predicted_latlon[valid_mask])
    metrics["top1_distance_error_m_mean"] = float(distances.mean().item())
    metrics["top1_distance_error_m_median"] = float(distances.median().item())
    for threshold in thresholds_m:
        key = f"top1_within_{int(threshold)}m"
        metrics[key] = float((distances <= threshold).float().mean().item())
    return metrics


def tile_masked_ranking(
    similarity: torch.Tensor,
    query_tiles: Sequence[str],
    gallery_tiles: Sequence[str],
) -> tuple[torch.Tensor, list[int]]:
    restricted_rows = []
    candidate_counts: list[int] = []
    for row_index, tile_name in enumerate(query_tiles):
        tile_mask = torch.tensor(
            [gallery_tile == tile_name and tile_name != "" for gallery_tile in gallery_tiles],
            device=similarity.device,
            dtype=torch.bool,
        )
        candidate_count = int(tile_mask.sum().item())
        candidate_counts.append(candidate_count)
        if candidate_count == 0:
            restricted_rows.append(similarity[row_index])
            continue
        masked = similarity[row_index].masked_fill(~tile_mask, float("-inf"))
        restricted_rows.append(masked)
    restricted_similarity = torch.stack(restricted_rows, dim=0)
    ranking = torch.argsort(restricted_similarity, dim=1, descending=True)
    return ranking, candidate_counts


def multi_tile_masked_ranking(
    similarity: torch.Tensor,
    predicted_tiles: Sequence[Sequence[str]],
    gallery_tiles: Sequence[str],
) -> tuple[torch.Tensor, list[int]]:
    gallery_indices_by_tile: dict[str, list[int]] = {}
    for gallery_index, tile_name in enumerate(gallery_tiles):
        if tile_name == "":
            continue
        gallery_indices_by_tile.setdefault(tile_name, []).append(gallery_index)

    restricted_rows = []
    candidate_counts: list[int] = []
    gallery_width = similarity.size(1)
    device = similarity.device

    for row_index, tile_names in enumerate(predicted_tiles):
        allowed_indices: list[int] = []
        seen_tiles: set[str] = set()
        for tile_name in tile_names:
            if tile_name == "" or tile_name in seen_tiles:
                continue
            seen_tiles.add(tile_name)
            allowed_indices.extend(gallery_indices_by_tile.get(tile_name, []))

        candidate_count = len(allowed_indices)
        candidate_counts.append(candidate_count)
        if candidate_count == 0:
            restricted_rows.append(similarity[row_index])
            continue

        mask = torch.zeros(gallery_width, device=device, dtype=torch.bool)
        mask[torch.tensor(allowed_indices, device=device, dtype=torch.long)] = True
        masked = similarity[row_index].masked_fill(~mask, float("-inf"))
        restricted_rows.append(masked)

    restricted_similarity = torch.stack(restricted_rows, dim=0)
    ranking = torch.argsort(restricted_similarity, dim=1, descending=True)
    return ranking, candidate_counts


def binary_classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    targets = targets.float()
    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def classification_predictions(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1):
        probabilities = torch.sigmoid(logits.reshape(-1))
        predictions = (probabilities >= 0.5).long()
        return predictions, probabilities
    probabilities = torch.softmax(logits, dim=-1)
    predictions = probabilities.argmax(dim=-1)
    return predictions.long(), probabilities


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    predictions, probabilities = classification_predictions(logits)
    targets = targets.long().reshape(-1)
    predictions = predictions.long().reshape(-1)

    if logits.ndim == 1 or (logits.ndim == 2 and logits.size(-1) == 1):
        num_classes = 2
    else:
        num_classes = int(logits.size(-1))
    if len(targets):
        num_classes = max(num_classes, int(targets.max().item()) + 1, int(predictions.max().item()) + 1)

    if class_names is None:
        resolved_class_names = [str(index) for index in range(num_classes)]
    else:
        resolved_class_names = list(class_names)
        if len(resolved_class_names) < num_classes:
            resolved_class_names.extend(str(index) for index in range(len(resolved_class_names), num_classes))

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets, predictions):
        if 0 <= int(target) < num_classes and 0 <= int(prediction) < num_classes:
            confusion[int(target), int(prediction)] += 1

    total = int(confusion.sum().item())
    accuracy = _safe_divide(float(confusion.diag().sum().item()), float(total))

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    weighted_f1_sum = 0.0
    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    supported_class_count = 0

    for class_index in range(num_classes):
        tp = float(confusion[class_index, class_index].item())
        fp = float(confusion[:, class_index].sum().item() - tp)
        fn = float(confusion[class_index, :].sum().item() - tp)
        support = float(confusion[class_index, :].sum().item())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        class_name = resolved_class_names[class_index]
        per_class[class_name] = {
            "support": int(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if support > 0:
            supported_class_count += 1
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            weighted_precision_sum += precision * support
            weighted_recall_sum += recall * support
            weighted_f1_sum += f1 * support

    macro_precision = _safe_divide(sum(precision_values), float(supported_class_count))
    macro_recall = _safe_divide(sum(recall_values), float(supported_class_count))
    macro_f1 = _safe_divide(sum(f1_values), float(supported_class_count))
    weighted_precision = _safe_divide(weighted_precision_sum, float(total))
    weighted_recall = _safe_divide(weighted_recall_sum, float(total))
    weighted_f1 = _safe_divide(weighted_f1_sum, float(total))

    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "num_classes": num_classes,
        "class_names": resolved_class_names[:num_classes],
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
    if num_classes == 2:
        positive_name = resolved_class_names[1]
        metrics["positive_class"] = positive_name
        metrics["positive_precision"] = per_class[positive_name]["precision"]
        metrics["positive_recall"] = per_class[positive_name]["recall"]
        metrics["positive_f1"] = per_class[positive_name]["f1"]
    if probabilities.ndim == 1:
        metrics["mean_positive_probability"] = float(probabilities.mean().item()) if len(probabilities) else 0.0
    return metrics


def psnr(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    mse = torch.mean((predictions - targets) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def dataset_psnr(total_squared_error: float, total_elements: int) -> float:
    if total_elements <= 0:
        return float("nan")
    mean_mse = total_squared_error / float(total_elements)
    if mean_mse <= 1e-12:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(1.0 / mean_mse)).item()
