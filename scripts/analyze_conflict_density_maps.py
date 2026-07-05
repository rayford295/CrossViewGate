from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conflict density as an unsupervised spatial damage signal, evaluated "
            "against an uncertainty-density control. Tiles are a lat/lon grid; "
            "per-sample street/remote probabilities are averaged across seeds "
            "before computing tile signals. Moran's I (permutation p) quantifies "
            "spatial autocorrelation of each signal."
        )
    )
    parser.add_argument("--multiseed-root", default="outputs/multiseed_main")
    parser.add_argument("--datasets", default="altadena_3class,ian_original,milton_original")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument(
        "--split-dirs",
        default=(
            "altadena_3class=data/splits/altadena_3class_objectid,"
            "ian_original=data/splits/ian_hurricane_original,"
            "milton_original=data/splits/milton_hurricane_original"
        ),
    )
    parser.add_argument("--grid-degrees", type=float, default=0.005)
    parser.add_argument(
        "--tile-col",
        default=None,
        help="Use this split-CSV column as the tile id instead of the lat/lon grid.",
    )
    parser.add_argument("--min-tile-size", type=int, default=5)
    parser.add_argument("--moran-permutations", type=int, default=9999)
    parser.add_argument("--output-dir", default="outputs/analysis/conflict_density_maps")
    parser.add_argument("--doc-path", default="docs/conflict_density_maps.md")
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


def _seed_mean_probs(root: Path, dataset: str, mode: str, seeds: list[int]) -> pd.DataFrame:
    """Per-sample probabilities averaged across seeds."""
    per_seed: list[pd.DataFrame] = []
    for seed in seeds:
        df = pd.read_csv(root / dataset / f"{mode}_seed{seed}" / "test_predictions.csv")
        df["sample_id"] = df["sample_id"].map(_canonical_sample_id)
        prob_cols = _numbered_columns(df, "prob")
        per_seed.append(df[["sample_id"] + prob_cols].set_index("sample_id"))
    stacked = pd.concat(per_seed).groupby(level=0).mean()
    return stacked


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = (p + q) / 2.0

    def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * (np.log(np.clip(a, 1e-12, 1.0)) - np.log(np.clip(b, 1e-12, 1.0)))).sum(axis=1)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def morans_i(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, permutations: int, seed: int = 42) -> tuple[float, float]:
    """Moran's I with inverse-distance weights and permutation p-value."""
    n = len(values)
    if n < 3:
        return float("nan"), float("nan")
    coords = np.stack([lat, lon], axis=1)
    distance = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    with np.errstate(divide="ignore"):
        weights = 1.0 / distance
    np.fill_diagonal(weights, 0.0)
    weights[~np.isfinite(weights)] = 0.0
    w_sum = weights.sum()
    centered = values - values.mean()
    denom = (centered**2).sum()
    if denom <= 0 or w_sum <= 0:
        return float("nan"), float("nan")

    def statistic(x: np.ndarray) -> float:
        c = x - x.mean()
        return float(n / w_sum * (weights * np.outer(c, c)).sum() / (c**2).sum())

    observed = statistic(values)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        if statistic(rng.permutation(values)) >= observed:
            count += 1
    p_value = (count + 1) / (permutations + 1)
    return observed, float(p_value)


def analyze_dataset(
    root: Path,
    dataset: str,
    split_dir: Path,
    seeds: list[int],
    grid_degrees: float,
    min_tile_size: int,
    moran_permutations: int,
    tile_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    split = pd.read_csv(split_dir / "test.csv")
    split["sample_id"] = split["sample_id"].map(_canonical_sample_id)
    street = _seed_mean_probs(root, dataset, "street_only", seeds)
    remote = _seed_mean_probs(root, dataset, "remote_only", seeds)
    common = street.index.intersection(remote.index)
    street = street.loc[common]
    remote = remote.loc[common]

    frame = split.set_index("sample_id").loc[common, ["latitude", "longitude", "label"]].copy()
    street_probs = street.to_numpy(dtype=np.float64)
    remote_probs = remote.to_numpy(dtype=np.float64)
    frame["conflict_hard"] = (street_probs.argmax(axis=1) != remote_probs.argmax(axis=1)).astype(float)
    frame["conflict_soft"] = _js_divergence(street_probs, remote_probs)
    frame["uncertainty"] = (_entropy(street_probs) + _entropy(remote_probs)) / 2.0
    frame["street_uncertainty"] = _entropy(street_probs)
    frame["remote_uncertainty"] = _entropy(remote_probs)

    if tile_col:
        frame[tile_col] = split.set_index("sample_id").loc[common, tile_col]
        frame = frame[frame[tile_col].notna()].copy()
        frame["tile_lat"] = frame.groupby(tile_col)["latitude"].transform("mean")
        frame["tile_lon"] = frame.groupby(tile_col)["longitude"].transform("mean")
        group_cols = [tile_col]
    else:
        frame["tile_lat"] = np.floor(frame["latitude"] / grid_degrees) * grid_degrees
        frame["tile_lon"] = np.floor(frame["longitude"] / grid_degrees) * grid_degrees
        group_cols = ["tile_lat", "tile_lon"]
    tiles = (
        frame.groupby(group_cols)
        .agg(
            n=("label", "count"),
            damage=("label", "mean"),
            conflict_hard=("conflict_hard", "mean"),
            conflict_soft=("conflict_soft", "mean"),
            uncertainty=("uncertainty", "mean"),
            street_uncertainty=("street_uncertainty", "mean"),
            remote_uncertainty=("remote_uncertainty", "mean"),
            lat=("tile_lat", "mean"),
            lon=("tile_lon", "mean"),
        )
        .query(f"n >= {min_tile_size}")
        .reset_index()
    )

    signals = ["conflict_hard", "conflict_soft", "uncertainty", "street_uncertainty", "remote_uncertainty"]
    summary: dict[str, object] = {"dataset": dataset, "n_tiles": int(len(tiles)), "grid_degrees": grid_degrees}
    for signal in signals:
        r, p = spearmanr(tiles[signal], tiles["damage"])
        summary[f"{signal}_spearman_r"] = float(r)
        summary[f"{signal}_spearman_p"] = float(p)
    lat = tiles["lat"].to_numpy()
    lon = tiles["lon"].to_numpy()
    for signal in ("damage", "conflict_soft"):
        moran, moran_p = morans_i(tiles[signal].to_numpy(dtype=np.float64), lat, lon, moran_permutations)
        summary[f"{signal}_moran_i"] = moran
        summary[f"{signal}_moran_p"] = moran_p
    return tiles, summary


def main() -> None:
    args = parse_args()
    root = Path(args.multiseed_root)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    split_dirs = dict(item.split("=", 1) for item in args.split_dirs.split(","))

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    summaries: list[dict[str, object]] = []
    for dataset in datasets:
        tiles, summary = analyze_dataset(
            root,
            dataset,
            Path(split_dirs[dataset]),
            seeds,
            args.grid_degrees,
            args.min_tile_size,
            args.moran_permutations,
            tile_col=args.tile_col,
        )
        tiles.to_csv(output_dir / f"tiles_{dataset}.csv", index=False)
        summaries.append(summary)
        print(f"analyzed {dataset}: {summary['n_tiles']} tiles")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "conflict_density_summary.csv", index=False)

    lines = [
        "# Conflict Density as an Unsupervised Spatial Damage Signal",
        "",
        "Tiles are a lat/lon grid (default 0.005 deg, roughly 500 m). Street and",
        "remote probabilities are averaged across seeds before computing signals.",
        "`conflict_soft` is the mean street-remote JS divergence per tile;",
        "`uncertainty` is the mean single-view entropy control. If conflict density",
        "correlates with tile damage better than plain uncertainty, the map carries",
        "cross-view information rather than repackaged model confidence.",
        "",
        "| dataset | tiles | conflict_soft r (p) | conflict_hard r (p) | uncertainty r (p) | damage Moran I (p) | conflict Moran I (p) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summaries:
        def rp(signal: str) -> str:
            return f"{row[f'{signal}_spearman_r']:.3f} ({row[f'{signal}_spearman_p']:.4f})"

        lines.append(
            f"| {row['dataset']} | {row['n_tiles']} | {rp('conflict_soft')} | {rp('conflict_hard')} | "
            f"{rp('uncertainty')} | {row['damage_moran_i']:.3f} ({row['damage_moran_p']:.4f}) | "
            f"{row['conflict_soft_moran_i']:.3f} ({row['conflict_soft_moran_p']:.4f}) |"
        )
    doc_path = Path(args.doc_path)
    ensure_dir(doc_path.parent)
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {doc_path}")
    print(summary_df.filter(regex="dataset|spearman_r").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
