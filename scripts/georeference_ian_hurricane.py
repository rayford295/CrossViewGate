from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.data.cvian import (
    CVIAN_CHECKSUM_RELATIVE_PATH,
    CVIAN_DOI,
    CVIAN_POSITION_RELATIVE_PATH,
    CVIAN_POSITION_SHA512,
    build_cvi_an_image_id_map,
    georeference_cvi_an_pairs,
    load_cvi_an_positions,
    local_position_geojson,
    sha512_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover official CVIAN image ids by exact SHA-512, join the official "
            "position GeoJSON, and write non-destructive georeferenced sidecars."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pairs-csv", help="Defaults to <dataset-root>/pairs.csv.")
    parser.add_argument(
        "--position-geojson",
        help="Defaults to <dataset-root>/02_Position/CVIAN_position.geojson.",
    )
    parser.add_argument(
        "--checksums",
        help="Official checksums.sha512. Defaults to <dataset-root>/checksums.sha512.",
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to <dataset-root>/georeferenced; raw CSV files are never overwritten.",
    )
    parser.add_argument("--grid-degrees", type=float, default=0.005)
    parser.add_argument(
        "--expected-position-sha512",
        default=CVIAN_POSITION_SHA512,
        help="Expected official GeoJSON checksum; pass an empty value only for a documented new version.",
    )
    return parser.parse_args()


def _sample_id(frame: pd.DataFrame) -> pd.Series:
    sat = frame["sat_path"].map(lambda value: Path(str(value)).stem.removesuffix("_sat"))
    svi = frame["svi_path"].map(lambda value: Path(str(value)).stem.removesuffix("_svi"))
    if not sat.equals(svi):
        raise ValueError("sat_path and svi_path contain different local sample ids")
    return sat.astype(str)


def _source_split_map(dataset_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split_name, filename in (("train", "train_split.csv"), ("test", "test_split.csv")):
        path = dataset_root / filename
        split = pd.read_csv(path)
        split["sample_id"] = _sample_id(split)
        split["original_split"] = split_name
        rows.append(split[["sample_id", "original_split"]])
    combined = pd.concat(rows, ignore_index=True)
    if combined["sample_id"].duplicated().any():
        duplicates = combined.loc[combined["sample_id"].duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"Samples occur in multiple source splits: {duplicates}")
    return combined


def _write_enriched_split(
    source_path: Path,
    enriched_pairs: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    source = pd.read_csv(source_path)
    source["sample_id"] = _sample_id(source)
    metadata_columns = [
        column
        for column in enriched_pairs.columns
        if column not in {"sat_path", "svi_path", "severity", "sample_id"}
    ]
    output = source.merge(
        enriched_pairs[["sample_id"] + metadata_columns],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    if output["latitude"].isna().any():
        raise ValueError(f"Incomplete georeference join while writing {output_path}")
    output.to_csv(output_path, index=False)
    return output


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    pairs_path = Path(args.pairs_csv).resolve() if args.pairs_csv else dataset_root / "pairs.csv"
    position_path = (
        Path(args.position_geojson).resolve()
        if args.position_geojson
        else dataset_root / CVIAN_POSITION_RELATIVE_PATH
    )
    checksums_path = (
        Path(args.checksums).resolve()
        if args.checksums
        else dataset_root / CVIAN_CHECKSUM_RELATIVE_PATH
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_root / "georeferenced"
    )

    for required in (pairs_path, position_path, checksums_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    position_sha512 = sha512_file(position_path)
    expected_sha512 = str(args.expected_position_sha512).strip().lower()
    if expected_sha512 and position_sha512 != expected_sha512:
        raise ValueError(
            f"Position GeoJSON SHA-512 mismatch: expected {expected_sha512}, got {position_sha512}"
        )

    pairs = pd.read_csv(pairs_path)
    required_pair_columns = {"sat_path", "svi_path", "severity"}
    missing_columns = required_pair_columns - set(pairs.columns)
    if missing_columns:
        raise KeyError(f"pairs CSV is missing columns: {sorted(missing_columns)}")
    positions = load_cvi_an_positions(position_path)
    image_id_map = build_cvi_an_image_id_map(dataset_root, pairs, checksums_path)
    enriched = georeference_cvi_an_pairs(
        pairs,
        image_id_map,
        positions,
        grid_degrees=args.grid_degrees,
    )
    split_map = _source_split_map(dataset_root)
    enriched = enriched.merge(split_map, on="sample_id", how="left", validate="one_to_one")
    if enriched["original_split"].isna().any():
        missing = enriched.loc[enriched["original_split"].isna(), "sample_id"].head().tolist()
        raise ValueError(f"Samples missing from original train/test split: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_id_map.to_csv(output_dir / "image_id_map.csv", index=False)
    enriched.to_csv(output_dir / "pairs_georeferenced.csv", index=False)
    train = _write_enriched_split(
        dataset_root / "train_split.csv",
        enriched,
        output_dir / "train_split_georeferenced.csv",
    )
    test = _write_enriched_split(
        dataset_root / "test_split.csv",
        enriched,
        output_dir / "test_split_georeferenced.csv",
    )
    (output_dir / "CVIAN_position_local.geojson").write_text(
        json.dumps(local_position_geojson(enriched), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "dataset": "CVIAN",
        "source_doi": CVIAN_DOI,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mapping_method": "exact local image SHA-512 to official checksums.sha512",
        "position_geojson_sha512": position_sha512,
        "expected_position_geojson_sha512": expected_sha512 or None,
        "position_feature_count": int(len(positions)),
        "pair_count": int(len(enriched)),
        "image_count_verified": int(len(enriched) * 2),
        "unique_mapillary_ids": int(enriched["mapillary_id"].nunique()),
        "unique_sequence_ids": int(enriched["sequence_id"].nunique()),
        "unique_spatial_blocks": int(enriched["spatial_block_id"].nunique()),
        "grid_degrees": float(args.grid_degrees),
        "original_split_counts": {"train": int(len(train)), "test": int(len(test))},
        "severity_counts": {
            str(key): int(value) for key, value in enriched["severity"].value_counts().items()
        },
        "bbox": {
            "min_longitude": float(enriched["longitude"].min()),
            "min_latitude": float(enriched["latitude"].min()),
            "max_longitude": float(enriched["longitude"].max()),
            "max_latitude": float(enriched["latitude"].max()),
        },
    }
    (output_dir / "georeference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote georeferenced sidecars to {output_dir}")


if __name__ == "__main__":
    main()
