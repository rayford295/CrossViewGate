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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the cross-view conflict gain across FOV variants: original "
            "panorama (from the main multiseed suite), building-centered crop, and "
            "random-centered crop. remote_only comes from the main suite in all "
            "variants; only the street view changes, so differences in the conflict "
            "gain are attributable to street-view target alignment."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--fov-root", default="outputs/fov_intervention")
    parser.add_argument("--datasets", default="ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--output-dir", default="outputs/analysis/fov_intervention")
    parser.add_argument("--doc-path", default="docs/fov_intervention_results.md")
    return parser.parse_args()


def _canonical_sample_id(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
    return df[["sample_id", "target", "prediction"]]


def analyze(
    dataset: str,
    variant: str,
    seed: int,
    street_path: Path,
    crossview_path: Path,
    remote_path: Path,
) -> dict[str, object]:
    street = _load(street_path).rename(columns={"prediction": "street_pred"})
    remote = _load(remote_path).rename(columns={"prediction": "remote_pred"})[["sample_id", "remote_pred"]]
    crossview = _load(crossview_path).rename(columns={"prediction": "crossview_pred"})[
        ["sample_id", "crossview_pred"]
    ]
    merged = street.merge(remote, on="sample_id").merge(crossview, on="sample_id")
    target = merged["target"].to_numpy(dtype=np.int64)
    street_pred = merged["street_pred"].to_numpy(dtype=np.int64)
    remote_pred = merged["remote_pred"].to_numpy(dtype=np.int64)
    crossview_pred = merged["crossview_pred"].to_numpy(dtype=np.int64)

    conflict = street_pred != remote_pred
    street_correct = street_pred == target
    remote_correct = remote_pred == target
    crossview_correct = crossview_pred == target
    best_single = max(float(street_correct[conflict].mean()), float(remote_correct[conflict].mean()))
    oracle = float((street_correct | remote_correct)[conflict].mean())
    crossview_conflict = float(crossview_correct[conflict].mean())
    gap = oracle - best_single
    return {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "n": int(len(merged)),
        "street_accuracy": float(street_correct.mean()),
        "crossview_accuracy": float(crossview_correct.mean()),
        "conflict_rate": float(conflict.mean()),
        "conflict_n": int(conflict.sum()),
        "street_conflict_accuracy": float(street_correct[conflict].mean()),
        "remote_conflict_accuracy": float(remote_correct[conflict].mean()),
        "best_single_conflict_accuracy": best_single,
        "crossview_conflict_accuracy": crossview_conflict,
        "oracle_conflict_accuracy": oracle,
        "conflict_gain": crossview_conflict - best_single,
        "oracle_gap_closure": (crossview_conflict - best_single) / gap if gap > 1e-9 else None,
    }


def main() -> None:
    args = parse_args()
    multiseed_root = Path(args.multiseed_root)
    fov_root = Path(args.fov_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            remote_path = multiseed_root / dataset / f"remote_only_seed{seed}" / "test_predictions.csv"
            rows.append(
                analyze(
                    dataset,
                    "original_panorama",
                    seed,
                    multiseed_root / dataset / f"street_only_seed{seed}" / "test_predictions.csv",
                    multiseed_root / dataset / f"crossview_seed{seed}" / "test_predictions.csv",
                    remote_path,
                )
            )
            for variant in ("building", "random"):
                run_root = fov_root / f"{dataset}_fov_{variant}"
                street_path = run_root / f"street_only_seed{seed}" / "test_predictions.csv"
                crossview_path = run_root / f"crossview_seed{seed}" / "test_predictions.csv"
                if not street_path.exists() or not crossview_path.exists():
                    print(f"skipping incomplete {dataset} {variant} seed={seed}")
                    continue
                rows.append(
                    analyze(dataset, f"fov_{variant}", seed, street_path, crossview_path, remote_path)
                )

    results = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    results.to_csv(output_dir / "fov_intervention_raw.csv", index=False)

    numeric = [
        "street_accuracy",
        "crossview_accuracy",
        "conflict_rate",
        "street_conflict_accuracy",
        "remote_conflict_accuracy",
        "crossview_conflict_accuracy",
        "conflict_gain",
        "oracle_gap_closure",
    ]
    summary = results.groupby(["dataset", "variant"])[numeric].agg(["mean", "std"])
    summary.to_csv(output_dir / "fov_intervention_summary.csv")

    lines = [
        "# FOV Intervention Results (Phase 2)",
        "",
        "Causal test of the view-regime claim: hurricane panoramas are cropped to a",
        "256x256 (90-degree) window that is either centered on the SegFormer building",
        "centroid (`fov_building`) or on a uniform random azimuth (`fov_random`).",
        "Geometry is identical between the two variants; only target alignment",
        "differs. `original_panorama` rows are the main multiseed suite. remote_only",
        "is shared across variants, so conflict subsets change only through the",
        "street model.",
        "",
        "| dataset | variant | street_acc | crossview_acc | conflict_rate | conflict_gain | gap_closure |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (dataset, variant), group in results.groupby(["dataset", "variant"]):
        def fmt(column: str) -> str:
            return f"{group[column].mean():.4f} +/- {group[column].std():.4f}"
        lines.append(
            f"| {dataset} | {variant} | {fmt('street_accuracy')} | {fmt('crossview_accuracy')} | "
            f"{fmt('conflict_rate')} | {fmt('conflict_gain')} | {fmt('oracle_gap_closure')} |"
        )
    doc_path = Path(args.doc_path)
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc_path}")
    print(results.groupby(["dataset", "variant"])[["conflict_gain", "oracle_gap_closure"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
