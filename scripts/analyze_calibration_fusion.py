from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir

MODES = ["street_only", "remote_only", "concat", "crossview"]
PREFIXES = {"street_only": "street", "remote_only": "remote", "concat": "concat", "crossview": "crossview"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose fusion gains into calibration vs complementary information. "
            "Fits per-view temperature on val predictions, re-evaluates fusion "
            "baselines on calibrated test probabilities, and reports how much of "
            "the oracle single-view gap each method closes on the conflict subset."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--datasets", default="altadena_3class,ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--num-bins", type=int, default=15)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", default="outputs/analysis/calibration_fusion")
    parser.add_argument("--doc-path", default="docs/calibration_decomposition.md")
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


def _load_split(path: Path) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    logit_cols = _numbered_columns(df, "logit")
    if not logit_cols:
        raise KeyError(f"No multiclass logit columns in {path}; calibration analysis needs logits.")
    logits = df[logit_cols].to_numpy(dtype=np.float64)
    targets = df["target"].to_numpy(dtype=np.int64)
    return logits, targets, df["sample_id"]


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


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, num_bins: int) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == targets).astype(np.float64)
    bins = np.clip((confidence * num_bins).astype(int), 0, num_bins - 1)
    ece = 0.0
    for bin_index in range(num_bins):
        mask = bins == bin_index
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


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


def mcnemar_p_value(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    """Two-sided exact McNemar test on paired correctness vectors."""
    b = int(((correct_a == 1) & (correct_b == 0)).sum())
    c = int(((correct_a == 0) & (correct_b == 1)).sum())
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    from math import comb

    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def paired_bootstrap_delta_ci(
    correct_a: np.ndarray, correct_b: np.ndarray, num_rounds: int, seed: int
) -> tuple[float, float]:
    """95% CI for mean(correct_a) - mean(correct_b) with paired resampling."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    deltas = np.empty(num_rounds)
    for round_index in range(num_rounds):
        idx = rng.integers(0, n, size=n)
        deltas[round_index] = correct_a[idx].mean() - correct_b[idx].mean()
    return float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def analyze_seed(
    root: Path, dataset: str, seed: int, num_bins: int, num_bootstrap: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    logits: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, np.ndarray] = {}
    sample_ids: dict[str, pd.Series] = {}
    for mode in MODES:
        run_dir = root / dataset / f"{mode}_seed{seed}"
        for split in ("val", "test"):
            split_logits, split_targets, split_ids = _load_split(run_dir / f"{split}_predictions.csv")
            logits.setdefault(mode, {})[split] = split_logits
            key = f"{mode}_{split}"
            targets[key] = split_targets
            sample_ids[key] = split_ids

    for mode in MODES[1:]:
        for split in ("val", "test"):
            if not sample_ids[f"{mode}_{split}"].equals(sample_ids[f"{MODES[0]}_{split}"]):
                raise ValueError(f"sample_id order mismatch for {dataset} seed={seed} {mode} {split}")

    test_target = targets["street_only_test"]
    temperature: dict[str, float] = {}
    calibration_rows: list[dict[str, object]] = []
    calibrated_probs: dict[str, np.ndarray] = {}
    raw_probs: dict[str, np.ndarray] = {}
    for mode in MODES:
        t = fit_temperature(logits[mode]["val"], targets[f"{mode}_val"])
        temperature[mode] = t
        raw = _softmax(logits[mode]["test"])
        calibrated = _softmax(logits[mode]["test"] / t)
        raw_probs[mode] = raw
        calibrated_probs[mode] = calibrated
        calibration_rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "mode": mode,
                "temperature": t,
                "ece_raw": expected_calibration_error(raw, test_target, num_bins),
                "ece_calibrated": expected_calibration_error(calibrated, test_target, num_bins),
                "nll_raw": _nll(logits[mode]["test"], test_target, 1.0),
                "nll_calibrated": _nll(logits[mode]["test"], test_target, t),
            }
        )

    street_pred = raw_probs["street_only"].argmax(axis=1)
    remote_pred = raw_probs["remote_only"].argmax(axis=1)
    conflict_mask = street_pred != remote_pred

    street_correct = street_pred == test_target
    remote_correct = remote_pred == test_target
    oracle_correct = street_correct | remote_correct
    best_single_conflict = max(
        float(street_correct[conflict_mask].mean()), float(remote_correct[conflict_mask].mean())
    )
    oracle_conflict = float(oracle_correct[conflict_mask].mean())
    oracle_gap = oracle_conflict - best_single_conflict

    def fusion_predictions(calibrated: bool) -> dict[str, np.ndarray]:
        street = calibrated_probs["street_only"] if calibrated else raw_probs["street_only"]
        remote = calibrated_probs["remote_only"] if calibrated else raw_probs["remote_only"]
        street_logits = logits["street_only"]["test"] / (temperature["street_only"] if calibrated else 1.0)
        remote_logits = logits["remote_only"]["test"] / (temperature["remote_only"] if calibrated else 1.0)
        prob_avg = ((street + remote) / 2.0).argmax(axis=1)
        logit_avg = ((street_logits + remote_logits) / 2.0).argmax(axis=1)
        street_argmax = street.argmax(axis=1)
        remote_argmax = remote.argmax(axis=1)
        voting = np.where(
            street_argmax == remote_argmax,
            street_argmax,
            np.where(street.max(axis=1) >= remote.max(axis=1), street_argmax, remote_argmax),
        )
        return {"probability_average": prob_avg, "logit_average": logit_avg, "confidence_voting": voting}

    method_predictions: dict[str, np.ndarray] = {
        "street_only": street_pred,
        "remote_only": remote_pred,
        "concat_reference": raw_probs["concat"].argmax(axis=1),
        "crossview_reference": raw_probs["crossview"].argmax(axis=1),
    }
    for name, prediction in fusion_predictions(calibrated=False).items():
        method_predictions[f"raw_{name}"] = prediction
    for name, prediction in fusion_predictions(calibrated=True).items():
        method_predictions[f"calibrated_{name}"] = prediction

    crossview_conflict_correct = (
        method_predictions["crossview_reference"][conflict_mask] == test_target[conflict_mask]
    ).astype(np.int64)

    method_rows: list[dict[str, object]] = []
    for method, prediction in method_predictions.items():
        correct = (prediction == test_target).astype(np.int64)
        conflict_correct = correct[conflict_mask]
        conflict_accuracy = float(conflict_correct.mean()) if conflict_mask.any() else None
        gap_closure = None
        if conflict_accuracy is not None and oracle_gap > 1e-9:
            gap_closure = (conflict_accuracy - best_single_conflict) / oracle_gap
        row: dict[str, object] = {
            "dataset": dataset,
            "seed": seed,
            "method": method,
            "accuracy": float(correct.mean()),
            "macro_f1": _macro_f1(test_target, prediction),
            "conflict_n": int(conflict_mask.sum()),
            "conflict_accuracy": conflict_accuracy,
            "best_single_conflict_accuracy": best_single_conflict,
            "oracle_conflict_accuracy": oracle_conflict,
            "oracle_gap_closure": gap_closure,
        }
        if method != "crossview_reference" and conflict_mask.any():
            delta = float(conflict_correct.mean() - crossview_conflict_correct.mean())
            ci_low, ci_high = paired_bootstrap_delta_ci(
                conflict_correct, crossview_conflict_correct, num_bootstrap, seed=seed
            )
            row.update(
                {
                    "conflict_delta_vs_crossview": delta,
                    "conflict_delta_ci_low": ci_low,
                    "conflict_delta_ci_high": ci_high,
                    "mcnemar_p_vs_crossview": mcnemar_p_value(conflict_correct, crossview_conflict_correct),
                }
            )
        method_rows.append(row)
    return calibration_rows, method_rows


def _summary_table(df: pd.DataFrame, value_columns: list[str], group_columns: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_columns)[value_columns]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std().add_suffix("_std")
    return mean.join(std).reset_index()


def _format_mean_std(mean: float, std: float) -> str:
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def write_doc(doc_path: Path, calibration: pd.DataFrame, methods: pd.DataFrame) -> None:
    lines: list[str] = [
        "# Calibration Decomposition (Phase 3)",
        "",
        "Per-view temperatures are fit on validation predictions and applied to test",
        "logits. Fusion baselines are then recomputed on calibrated probabilities to",
        "separate calibration effects from genuine complementary-information effects.",
        "Conflict subsets use raw single-view argmax disagreement (temperature scaling",
        "does not change single-view argmax, so the subset is identical before/after).",
        "`oracle_gap_closure` is (method - best single view) / (oracle single view -",
        "best single view) on the conflict subset.",
        "",
        "## Per-view calibration quality",
        "",
        "| dataset | mode | temperature | ECE raw | ECE calibrated |",
        "| --- | --- | --- | --- | --- |",
    ]
    calib_summary = _summary_table(
        calibration, ["temperature", "ece_raw", "ece_calibrated"], ["dataset", "mode"]
    )
    for _, row in calib_summary.iterrows():
        lines.append(
            "| {dataset} | {mode} | {t} | {raw} | {cal} |".format(
                dataset=row["dataset"],
                mode=row["mode"],
                t=_format_mean_std(row["temperature_mean"], row["temperature_std"]),
                raw=_format_mean_std(row["ece_raw_mean"], row["ece_raw_std"]),
                cal=_format_mean_std(row["ece_calibrated_mean"], row["ece_calibrated_std"]),
            )
        )
    lines += [
        "",
        "## Fusion methods, raw vs calibrated",
        "",
        "| dataset | method | accuracy | macro_f1 | conflict_acc | oracle_gap_closure |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    method_summary = _summary_table(
        methods,
        ["accuracy", "macro_f1", "conflict_accuracy", "oracle_gap_closure"],
        ["dataset", "method"],
    )
    for _, row in method_summary.iterrows():
        lines.append(
            "| {dataset} | {method} | {acc} | {f1} | {conf} | {gap} |".format(
                dataset=row["dataset"],
                method=row["method"],
                acc=_format_mean_std(row["accuracy_mean"], row["accuracy_std"]),
                f1=_format_mean_std(row["macro_f1_mean"], row["macro_f1_std"]),
                conf=_format_mean_std(row["conflict_accuracy_mean"], row["conflict_accuracy_std"]),
                gap=_format_mean_std(row["oracle_gap_closure_mean"], row["oracle_gap_closure_std"]),
            )
        )
    lines += [
        "",
        "## Paired comparison vs crossview on the conflict subset",
        "",
        "Per-seed paired bootstrap CIs and exact McNemar p-values against the",
        "crossview reference; `sig` counts seeds with p < 0.05.",
        "",
        "| dataset | method | delta vs crossview | McNemar sig |",
        "| --- | --- | --- | --- |",
    ]
    paired = methods.dropna(subset=["conflict_delta_vs_crossview"]) if "conflict_delta_vs_crossview" in methods else pd.DataFrame()
    if not paired.empty:
        for (dataset, method), group in paired.groupby(["dataset", "method"]):
            sig = int((group["mcnemar_p_vs_crossview"] < 0.05).sum())
            lines.append(
                "| {dataset} | {method} | {delta} | {sig}/{n} |".format(
                    dataset=dataset,
                    method=method,
                    delta=_format_mean_std(
                        group["conflict_delta_vs_crossview"].mean(),
                        group["conflict_delta_vs_crossview"].std(),
                    ),
                    sig=sig,
                    n=len(group),
                )
            )
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.multiseed_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    calibration_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            seed_calibration, seed_methods = analyze_seed(
                root, dataset, seed, args.num_bins, args.bootstrap
            )
            calibration_rows.extend(seed_calibration)
            method_rows.extend(seed_methods)
            print(f"analyzed {dataset} seed={seed}")

    calibration = pd.DataFrame(calibration_rows)
    methods = pd.DataFrame(method_rows)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    calibration.to_csv(output_dir / "calibration_raw.csv", index=False)
    methods.to_csv(output_dir / "fusion_methods_raw.csv", index=False)
    _summary_table(
        calibration, ["temperature", "ece_raw", "ece_calibrated", "nll_raw", "nll_calibrated"], ["dataset", "mode"]
    ).to_csv(output_dir / "calibration_summary.csv", index=False)
    _summary_table(
        methods,
        ["accuracy", "macro_f1", "conflict_accuracy", "oracle_gap_closure"],
        ["dataset", "method"],
    ).to_csv(output_dir / "fusion_methods_summary.csv", index=False)
    write_doc(Path(args.doc_path), calibration, methods)
    print(f"wrote {args.doc_path}")


if __name__ == "__main__":
    main()
