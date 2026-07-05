from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir

VISIBILITY_FEATURES = [
    "building_ratio",
    "center_building_ratio",
    "center_minus_global",
    "centroid_distance_norm",
]
DERIVED_FEATURES = [
    "street_confidence",
    "remote_confidence",
    "street_entropy",
    "remote_entropy",
    "confidence_gap",
    "js_divergence",
    "views_disagree",
]
ALL_FEATURES = VISIBILITY_FEATURES + DERIVED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visibility-conditioned reliability gate. Learns a per-sample street-vs-"
            "remote mixture weight from building visibility, calibrated confidence, "
            "and cross-view disagreement. Trains on val predictions, evaluates on "
            "test, and supports zero-shot cross-disaster transfer."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--visibility-dir", default="outputs/analysis/visibility_features")
    parser.add_argument("--datasets", default="altadena_3class,ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="outputs/analysis/reliability_gate")
    parser.add_argument("--doc-path", default="docs/reliability_gate_results.md")
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _numbered_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    pairs: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for column in df.columns:
        match = pattern.match(column)
        if match:
            pairs.append((int(match.group(1)), column))
    return [column for _, column in sorted(pairs)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(targets)), targets].mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
    losses = [_nll(logits, targets, float(t)) for t in grid]
    best = float(grid[int(np.argmin(losses))])
    low, high = best / 1.25, best * 1.25
    for _ in range(40):
        mid1 = low + (high - low) / 3.0
        mid2 = high - (high - low) / 3.0
        if _nll(logits, targets, mid1) < _nll(logits, targets, mid2):
            high = mid2
        else:
            low = mid1
    return float((low + high) / 2.0)


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2.0
    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * (np.log(np.clip(a, 1e-12, 1.0)) - np.log(np.clip(b, 1e-12, 1.0)))).sum(axis=1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class SplitData:
    """Aligned per-sample arrays for one dataset/seed/split."""

    def __init__(
        self,
        features: np.ndarray,
        street_probs: np.ndarray,
        remote_probs: np.ndarray,
        targets: np.ndarray,
        crossview_probs: np.ndarray | None = None,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        self.features = features
        self.street_probs = street_probs
        self.remote_probs = remote_probs
        self.targets = targets
        self.crossview_probs = crossview_probs
        self.sample_ids = sample_ids


def load_split(
    root: Path,
    visibility_dir: Path,
    dataset: str,
    seed: int,
    split: str,
    temperatures: dict[str, float],
) -> SplitData:
    frames: dict[str, pd.DataFrame] = {}
    for mode in ("street_only", "remote_only", "crossview"):
        path = root / dataset / f"{mode}_seed{seed}" / f"{split}_predictions.csv"
        df = pd.read_csv(path)
        df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
        frames[mode] = df
    visibility = pd.read_csv(visibility_dir / f"{dataset}_{split}.csv")
    visibility["sample_id"] = visibility["sample_id"].map(_canonical_sample_id)
    if "error" in visibility.columns:
        visibility = visibility[visibility["error"].isna()]

    crossview_logit_cols = _numbered_columns(frames["crossview"], "logit")
    crossview_frame = frames["crossview"][["sample_id"] + crossview_logit_cols].rename(
        columns={col: f"{col}_crossview" for col in crossview_logit_cols}
    )
    merged = frames["street_only"].merge(
        frames["remote_only"], on="sample_id", suffixes=("_street", "_remote")
    )
    merged = merged.merge(crossview_frame, on="sample_id", how="inner")
    merged = merged.merge(visibility[["sample_id"] + VISIBILITY_FEATURES], on="sample_id", how="inner")
    if merged.empty:
        raise ValueError(f"Empty merge for {dataset} seed={seed} {split}")

    street_logit_cols = [f"{c}_street" for c in _numbered_columns(frames["street_only"], "logit")]
    remote_logit_cols = [f"{c}_remote" for c in _numbered_columns(frames["remote_only"], "logit")]
    street_logits = merged[street_logit_cols].to_numpy(dtype=np.float64)
    remote_logits = merged[remote_logit_cols].to_numpy(dtype=np.float64)
    crossview_logits = merged[[f"{c}_crossview" for c in crossview_logit_cols]].to_numpy(dtype=np.float64)
    street_probs = _softmax(street_logits / temperatures["street_only"])
    remote_probs = _softmax(remote_logits / temperatures["remote_only"])
    crossview_probs = _softmax(crossview_logits / temperatures["crossview"])
    targets = merged["target_street"].to_numpy(dtype=np.int64)

    derived = pd.DataFrame(
        {
            "street_confidence": street_probs.max(axis=1),
            "remote_confidence": remote_probs.max(axis=1),
            "street_entropy": _entropy(street_probs),
            "remote_entropy": _entropy(remote_probs),
            "confidence_gap": street_probs.max(axis=1) - remote_probs.max(axis=1),
            "js_divergence": _js_divergence(street_probs, remote_probs),
            "views_disagree": (street_probs.argmax(axis=1) != remote_probs.argmax(axis=1)).astype(float),
        }
    )
    features = np.concatenate(
        [merged[VISIBILITY_FEATURES].to_numpy(dtype=np.float64), derived.to_numpy(dtype=np.float64)],
        axis=1,
    )
    return SplitData(
        features,
        street_probs,
        remote_probs,
        targets,
        crossview_probs,
        merged["sample_id"].to_numpy(),
    )


class GateModel(nn.Module):
    """num_views=2: sigmoid street weight. num_views>2: softmax mixture weights."""

    def __init__(self, num_features: int, hidden_dim: int, num_views: int = 2) -> None:
        super().__init__()
        self.num_views = num_views
        out_dim = 1 if num_views == 2 else num_views
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.net = nn.Linear(num_features, out_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Returns per-view weights of shape (N, num_views)."""
        raw = self.net(features)
        if self.num_views == 2:
            street = torch.sigmoid(raw).squeeze(-1)
            return torch.stack([street, 1.0 - street], dim=-1)
        return torch.softmax(raw, dim=-1)


def _view_stack(data: SplitData, num_views: int) -> np.ndarray:
    views = [data.street_probs, data.remote_probs]
    if num_views == 3:
        if data.crossview_probs is None:
            raise ValueError("crossview probs required for 3-view gate")
        views.append(data.crossview_probs)
    return np.stack(views, axis=1)  # (N, num_views, C)


def train_gate(
    train_data: SplitData,
    features_normalized: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    num_views: int = 2,
    sample_weights: np.ndarray | None = None,
) -> GateModel:
    torch.manual_seed(seed)
    features = torch.tensor(features_normalized, dtype=torch.float32)
    views = torch.tensor(_view_stack(train_data, num_views), dtype=torch.float32)
    targets = torch.tensor(train_data.targets, dtype=torch.long)
    weights = None
    if sample_weights is not None:
        weights = torch.tensor(sample_weights / sample_weights.mean(), dtype=torch.float32)
    model = GateModel(features.shape[1], hidden_dim, num_views)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        optimizer.zero_grad()
        view_weights = model(features).unsqueeze(-1)  # (N, num_views, 1)
        mixture = (view_weights * views).sum(dim=1)
        log_mixture = torch.log(torch.clamp(mixture, min=1e-12))
        per_sample = nn.functional.nll_loss(log_mixture, targets, reduction="none")
        loss = (per_sample * weights).mean() if weights is not None else per_sample.mean()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def apply_gate(
    model: GateModel, data: SplitData, features_normalized: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.tensor(features_normalized, dtype=torch.float32)
    with torch.no_grad():
        view_weights = model(features).numpy()  # (N, num_views)
    mixture = (view_weights[:, :, None] * _view_stack(data, model.num_views)).sum(axis=1)
    return mixture.argmax(axis=1), view_weights[:, 0]


def mcnemar_p_value(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def _macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    num_classes = int(max(target.max(initial=0), prediction.max(initial=0)) + 1)
    f1_values: list[float] = []
    for class_index in range(num_classes):
        tp = float(((prediction == class_index) & (target == class_index)).sum())
        fp = float(((prediction == class_index) & (target != class_index)).sum())
        fn = float(((prediction != class_index) & (target == class_index)).sum())
        support = float((target == class_index).sum())
        if support == 0:
            continue
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1_values.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(f1_values)) if f1_values else 0.0


def evaluate(
    dataset: str,
    seed: int,
    variant: str,
    train_source: str,
    prediction: np.ndarray,
    gate_weight: np.ndarray | None,
    test: SplitData,
    crossview_pred: np.ndarray,
) -> dict[str, object]:
    target = test.targets
    street_pred = test.street_probs.argmax(axis=1)
    remote_pred = test.remote_probs.argmax(axis=1)
    conflict_mask = street_pred != remote_pred
    street_correct = street_pred == target
    remote_correct = remote_pred == target
    oracle_conflict = float((street_correct | remote_correct)[conflict_mask].mean())
    best_single_conflict = max(
        float(street_correct[conflict_mask].mean()), float(remote_correct[conflict_mask].mean())
    )
    correct = (prediction == target).astype(np.int64)
    conflict_accuracy = float(correct[conflict_mask].mean())
    gap = oracle_conflict - best_single_conflict
    crossview_correct = (crossview_pred == target).astype(np.int64)
    row: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "train_source": train_source,
        "accuracy": float(correct.mean()),
        "macro_f1": _macro_f1(target, prediction),
        "conflict_n": int(conflict_mask.sum()),
        "conflict_accuracy": conflict_accuracy,
        "best_single_conflict_accuracy": best_single_conflict,
        "oracle_conflict_accuracy": oracle_conflict,
        "oracle_gap_closure": (conflict_accuracy - best_single_conflict) / gap if gap > 1e-9 else None,
        "mcnemar_p_vs_crossview_conflict": mcnemar_p_value(
            correct[conflict_mask], crossview_correct[conflict_mask]
        ),
        "delta_vs_crossview_conflict": float(
            correct[conflict_mask].mean() - crossview_correct[conflict_mask].mean()
        ),
    }
    if gate_weight is not None:
        row["mean_street_weight"] = float(gate_weight.mean())
        row["mean_street_weight_on_conflicts"] = float(gate_weight[conflict_mask].mean())
    return row


def main() -> None:
    args = parse_args()
    root = Path(args.multiseed_root)
    visibility_dir = Path(args.visibility_dir)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, int, str], SplitData] = {}
    crossview_preds: dict[tuple[str, int], np.ndarray] = {}

    for dataset in datasets:
        for seed in seeds:
            temperatures: dict[str, float] = {}
            for mode in ("street_only", "remote_only", "crossview"):
                val_path = root / dataset / f"{mode}_seed{seed}" / "val_predictions.csv"
                df = pd.read_csv(val_path)
                logit_cols = _numbered_columns(df, "logit")
                temperatures[mode] = fit_temperature(
                    df[logit_cols].to_numpy(dtype=np.float64), df["target"].to_numpy(dtype=np.int64)
                )
            for split in ("val", "test"):
                cache[(dataset, seed, split)] = load_split(
                    root, visibility_dir, dataset, seed, split, temperatures
                )
            crossview_df = pd.read_csv(root / dataset / f"crossview_seed{seed}" / "test_predictions.csv")
            crossview_df["sample_id"] = crossview_df["sample_id"].map(_canonical_sample_id)
            test_street = pd.read_csv(root / dataset / f"street_only_seed{seed}" / "test_predictions.csv")
            test_street["sample_id"] = test_street["sample_id"].map(_canonical_sample_id)
            aligned = test_street[["sample_id"]].merge(crossview_df, on="sample_id")
            crossview_preds[(dataset, seed)] = aligned["prediction"].to_numpy(dtype=np.int64)

    def _zscore(features: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mean = reference.mean(axis=0)
        std = np.clip(reference.std(axis=0), 1e-6, None)
        return (features - mean) / std

    def pooled_split(source_datasets: list[str], seed: int) -> tuple[SplitData, np.ndarray, np.ndarray]:
        parts = [cache[(name, seed, "val")] for name in source_datasets]
        data = SplitData(
            np.concatenate([part.features for part in parts]),
            np.concatenate([part.street_probs for part in parts]),
            np.concatenate([part.remote_probs for part in parts]),
            np.concatenate([part.targets for part in parts]),
            np.concatenate([part.crossview_probs for part in parts]),
        )
        znorm_features = np.concatenate([_zscore(part.features, part.features) for part in parts])
        balance = np.concatenate(
            [np.full(len(part.targets), 1.0 / len(part.targets)) for part in parts]
        )
        return data, znorm_features, balance

    variants = (("gate_mlp", args.hidden_dim), ("gate_linear", 0))

    for target_dataset in datasets:
        for seed in seeds:
            test = cache[(target_dataset, seed, "test")]
            crossview_pred = crossview_preds[(target_dataset, seed)]
            target_val = cache[(target_dataset, seed, "val")]
            other_datasets = [name for name in datasets if name != target_dataset]
            if other_datasets:
                pooled, pooled_znorm, balance = pooled_split(other_datasets, seed)
                source_label = "+".join(other_datasets)
                # Raw pooled: shared source statistics, unbalanced (reference).
                raw_normalized = _zscore(pooled.features, pooled.features)
                test_raw_normalized = _zscore(test.features, pooled.features)
                # Regime-normalized pooled: each source dataset z-scored with its
                # own statistics, balanced sampling; the target is normalized with
                # its own (label-free) val statistics.
                test_znorm = _zscore(test.features, target_val.features)
                for variant, hidden in variants:
                    model = train_gate(
                        pooled, raw_normalized, hidden, args.epochs, args.learning_rate,
                        args.weight_decay, seed,
                    )
                    prediction, weight = apply_gate(model, test, test_raw_normalized)
                    rows.append(
                        evaluate(
                            target_dataset, seed, f"{variant}_loo_pooled", source_label,
                            prediction, weight, test, crossview_pred,
                        )
                    )
                    model = train_gate(
                        pooled, pooled_znorm, hidden, args.epochs, args.learning_rate,
                        args.weight_decay, seed, sample_weights=balance,
                    )
                    prediction, weight = apply_gate(model, test, test_znorm)
                    rows.append(
                        evaluate(
                            target_dataset, seed, f"{variant}_loo_pooled_znorm", source_label,
                            prediction, weight, test, crossview_pred,
                        )
                    )
            for source_dataset in datasets:
                train = cache[(source_dataset, seed, "val")]
                train_normalized = _zscore(train.features, train.features)
                test_normalized = _zscore(test.features, train.features)
                scope = "in_domain" if source_dataset == target_dataset else "transfer"
                for variant, hidden in variants:
                    model = train_gate(
                        train, train_normalized, hidden, args.epochs, args.learning_rate,
                        args.weight_decay, seed,
                    )
                    prediction, weight = apply_gate(model, test, test_normalized)
                    rows.append(
                        evaluate(
                            target_dataset, seed, f"{variant}_{scope}", source_dataset,
                            prediction, weight, test, crossview_pred,
                        )
                    )
                    if variant == "gate_linear" and source_dataset == target_dataset:
                        linear = model.net
                        for name, value in zip(ALL_FEATURES, linear.weight.detach().numpy().ravel()):
                            weight_rows.append(
                                {"dataset": target_dataset, "seed": seed, "feature": name, "coefficient": float(value)}
                            )
                    if source_dataset == target_dataset:
                        if variant == "gate_linear":
                            two_view_prediction = prediction
                        model = train_gate(
                            train, train_normalized, hidden, args.epochs, args.learning_rate,
                            args.weight_decay, seed, num_views=3,
                        )
                        prediction, weight = apply_gate(model, test, test_normalized)
                        rows.append(
                            evaluate(
                                target_dataset, seed, f"gate3_{'mlp' if hidden else 'linear'}_in_domain",
                                source_dataset, prediction, weight, test, crossview_pred,
                            )
                        )
                        if variant == "gate_linear":
                            pred_dir = Path(args.output_dir) / "predictions"
                            ensure_dir(pred_dir)
                            pd.DataFrame(
                                {
                                    "sample_id": test.sample_ids,
                                    "target": test.targets,
                                    "gate_linear_prediction": two_view_prediction,
                                    "gate3_linear_prediction": prediction,
                                }
                            ).to_csv(pred_dir / f"{target_dataset}_seed{seed}.csv", index=False)
            print(f"gated {target_dataset} seed={seed}")

    results = pd.DataFrame(rows)
    coefficients = pd.DataFrame(weight_rows)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    results.to_csv(output_dir / "gate_results_raw.csv", index=False)
    coefficients.to_csv(output_dir / "gate_linear_coefficients.csv", index=False)

    numeric = ["accuracy", "macro_f1", "conflict_accuracy", "oracle_gap_closure", "delta_vs_crossview_conflict"]
    summary = (
        results.groupby(["dataset", "variant", "train_source"])[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary.to_csv(output_dir / "gate_results_summary.csv", index=False)

    lines = [
        "# Reliability Gate Results (Phase 1)",
        "",
        "Gate: per-sample street-vs-remote mixture weight predicted from building",
        "visibility, calibrated confidence/entropy, confidence gap, JS divergence,",
        "and a disagreement flag. Trained on val by minimizing the NLL of the gated",
        "mixture; evaluated on test. `transfer` rows train the gate on a different",
        "dataset's val split (zero-shot cross-disaster).",
        "",
        "| dataset | variant | train source | accuracy | macro_f1 | conflict_acc | gap closure | sig vs crossview |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (dataset, variant, source), group in results.groupby(["dataset", "variant", "train_source"]):
        sig = int((group["mcnemar_p_vs_crossview_conflict"] < 0.05).sum())
        def fmt(column: str) -> str:
            return f"{group[column].mean():.4f} +/- {group[column].std():.4f}"
        lines.append(
            f"| {dataset} | {variant} | {source} | {fmt('accuracy')} | {fmt('macro_f1')} | "
            f"{fmt('conflict_accuracy')} | {fmt('oracle_gap_closure')} | {sig}/{len(group)} |"
        )
    if not coefficients.empty:
        lines += [
            "",
            "## Linear gate coefficients (in-domain, positive = trust street view)",
            "",
            "| dataset | feature | coefficient |",
            "| --- | --- | --- |",
        ]
        coef_summary = coefficients.groupby(["dataset", "feature"])["coefficient"].agg(["mean", "std"]).reset_index()
        for _, row in coef_summary.iterrows():
            lines.append(f"| {row['dataset']} | {row['feature']} | {row['mean']:.4f} +/- {row['std']:.4f} |")
    doc_path = Path(args.doc_path)
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc_path}")


if __name__ == "__main__":
    main()
