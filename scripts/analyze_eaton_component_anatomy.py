#!/usr/bin/env python
"""RQ1 Study B: Eaton component-direction anatomy or a fail-closed feasibility audit.

The DINS join contains inspector-recorded reference attributes.  Those values
are not image-level observations and construction/material fields are not
component-damage labels.  This runner therefore executes H-B1/H-B2 only when
separate, provenance-bearing component-damage references and genuine Eaton
per-view severity predictions are supplied.  Otherwise it writes a machine-
readable availability and power audit with an explicit blocked status.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


WHITELIST_SCHEMA = "eaton-image-visible-field-whitelist-v1"
AUDIT_SCHEMA = "eaton-component-anatomy-audit-v1"
ALLOWED_VIEW_STATUSES = {"primary_conditional", "stress_test_only", "excluded"}
ALLOWED_DOMINANCE = {"roof", "facade", "mixed", "unknown"}
UNKNOWN_TOKENS = {"unknown", "not known", "undetermined"}
PREDICTION_COLUMNS = {
    "pair_id",
    "seed",
    "street_prediction",
    "remote_prediction",
    "spatial_block_id",
}
REFERENCE_COLUMNS = {
    "pair_id",
    "component_dominance",
    "reference_semantics",
    "annotation_provenance",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.casefold()


def load_and_validate_whitelist(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != WHITELIST_SCHEMA:
        raise ValueError(
            f"Expected whitelist schema {WHITELIST_SCHEMA!r}, got "
            f"{payload.get('schema_version')!r}"
        )
    fields = payload.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("Whitelist must contain a non-empty fields list")
    columns = [field.get("column") for field in fields]
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("Every whitelist field needs a non-empty column")
    if len(columns) != len(set(columns)):
        raise ValueError("Whitelist columns must be unique")
    declared_non_damage = set(payload.get("explicit_non_damage_fields", []))
    for field in fields:
        for view in ("street", "remote"):
            status = field.get(view, {}).get("status")
            if status not in ALLOWED_VIEW_STATUSES:
                raise ValueError(
                    f"Invalid {view} status {status!r} for {field['column']}"
                )
        if field.get("component_damage_reference") is not False:
            raise ValueError(
                f"Inspector construction field {field['column']} must be explicitly "
                "marked as not a component-damage reference"
            )
        if field["column"] not in declared_non_damage:
            raise ValueError(
                f"{field['column']} is missing from explicit_non_damage_fields"
            )
    proxy = payload.get("restricted_proxy", {})
    if proxy.get("column") != "dins_wherefirestartedonstructure":
        raise ValueError("Restricted fire-origin proxy must be declared explicitly")
    if "must not" not in str(proxy.get("forbidden_use", "")).casefold():
        raise ValueError("Restricted proxy needs an explicit forbidden-use rule")
    return payload


def _source_key(frame: pd.DataFrame) -> pd.Series:
    if "dins_source_globalid" in frame:
        key = frame["dins_source_globalid"].astype("string")
    elif "dins_source_objectid" in frame:
        key = frame["dins_source_objectid"].astype("string")
    else:
        key = pd.Series(pd.NA, index=frame.index, dtype="string")
    missing = key.isna() | key.str.strip().eq("")
    key = key.mask(missing, "unmatched:" + frame["pair_id"].astype(str))
    return key


def audit_reference_fields(
    frame: pd.DataFrame, whitelist: dict
) -> tuple[pd.DataFrame, dict]:
    required = {"pair_id", "dins_join_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Joined manifest missing columns: {sorted(missing)}")
    if frame["pair_id"].duplicated().any():
        raise ValueError("pair_id must be unique in the joined manifest")

    source_key = _source_key(frame)
    with_key = frame.assign(_source_key=source_key)
    unique_sources = with_key.drop_duplicates("_source_key")
    rows: list[dict] = []
    for field in whitelist["fields"]:
        column = field["column"]
        if column not in frame.columns:
            raise ValueError(f"Whitelisted field absent from joined manifest: {column}")
        row_values = _normalized_text(frame[column])
        source_values = _normalized_text(unique_sources[column])
        row_non_null = row_values.notna() & row_values.ne("")
        source_non_null = source_values.notna() & source_values.ne("")
        row_unknown = row_values.isin(UNKNOWN_TOKENS)
        source_unknown = source_values.isin(UNKNOWN_TOKENS)
        rows.append(
            {
                "field": column,
                "component": field["component"],
                "inspector_reference_semantics": field[
                    "inspector_reference_semantics"
                ],
                "component_damage_reference": False,
                "street_status": field["street"]["status"],
                "remote_status": field["remote"]["status"],
                "attachment_rows_non_null": int(row_non_null.sum()),
                "attachment_rows_unknown": int(row_unknown.sum()),
                "attachment_rows_usable_reference": int(
                    (row_non_null & ~row_unknown).sum()
                ),
                "unique_structures_non_null": int(source_non_null.sum()),
                "unique_structures_unknown": int(source_unknown.sum()),
                "unique_structures_usable_reference": int(
                    (source_non_null & ~source_unknown).sum()
                ),
                "unique_non_null_values": int(
                    frame.loc[row_non_null, column].nunique(dropna=True)
                ),
            }
        )

    matched = frame["dins_join_status"].astype(str).str.startswith("matched")
    category_agreement: dict[str, int | None]
    if {"category", "dins_damage"}.issubset(frame.columns):
        comparable = matched & frame["category"].notna() & frame["dins_damage"].notna()
        equal = _normalized_text(frame.loc[comparable, "category"]).eq(
            _normalized_text(frame.loc[comparable, "dins_damage"])
        )
        category_agreement = {
            "comparable_rows": int(comparable.sum()),
            "equal_rows": int(equal.sum()),
            "mismatch_rows": int((~equal).sum()),
        }
    else:
        category_agreement = {
            "comparable_rows": None,
            "equal_rows": None,
            "mismatch_rows": None,
        }

    source_counts = with_key["_source_key"].value_counts()
    component_tokens = ("roof", "eave", "siding", "window", "deck", "porch", "vent")
    damage_component_columns = [
        str(column)
        for column in frame.columns
        if "damage" in str(column).casefold()
        and any(token in str(column).casefold() for token in component_tokens)
    ]
    summary = {
        "attachment_rows": int(len(frame)),
        "unique_pair_ids": int(frame["pair_id"].nunique()),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "unique_source_structures_including_unmatched_placeholders": int(
            unique_sources["_source_key"].nunique()
        ),
        "source_structures_with_multiple_attachments": int((source_counts > 1).sum()),
        "maximum_attachments_per_source_structure": int(source_counts.max()),
        "overall_damage_label_agreement": category_agreement,
        "component_damage_reference_columns_found": damage_component_columns,
        "construct_conclusion": (
            "The whitelisted DINS columns are inspector-recorded construction, "
            "material, or exposure attributes. They are not component-damage "
            "presence or component-damage dominance labels."
        ),
    }
    return pd.DataFrame(rows), summary


def audit_restricted_origin_proxy(frame: pd.DataFrame) -> dict:
    column = "dins_wherefirestartedonstructure"
    if column not in frame.columns:
        return {"available": False, "reason": f"{column} absent"}
    unique_sources = frame.assign(_source_key=_source_key(frame)).drop_duplicates(
        "_source_key"
    )
    origin = unique_sources[column].astype("string").str.strip()
    damage = unique_sources.get(
        "dins_damage", pd.Series(pd.NA, index=unique_sources.index, dtype="string")
    ).astype("string").str.strip()
    affected = damage.eq("Affected (1-9%)")
    known = origin.notna() & ~_normalized_text(origin).isin(UNKNOWN_TOKENS)
    roof = affected & origin.eq("Roof")
    facade = affected & origin.isin(["Siding", "Window", "Eaves"])
    roof_n = int(roof.sum())
    facade_n = int(facade.sum())
    mde = best_case_two_proportion_mde(roof_n, facade_n)
    return {
        "available": True,
        "semantics": (
            "Inspector-recorded fire-origin location for Affected (1-9%) only; "
            "not component-damage dominance."
        ),
        "unique_structures": int(len(unique_sources)),
        "non_null_unique_structures": int(origin.notna().sum()),
        "known_affected_unique_structures": int((affected & known).sum()),
        "affected_roof_origin_unique_structures": roof_n,
        "affected_facade_origin_unique_structures": facade_n,
        "facade_definition_for_diagnostic_only": ["Siding", "Window", "Eaves"],
        "best_case_independent_two_proportion_mde_80pct_power": mde,
        "power_interpretation": (
            "Even if this invalid proxy were used and every structure yielded a "
            "disagreement, the roof-vs-facade direction-share contrast would need "
            f"to be about {mde:.3f} or larger under the stated normal approximation."
            if mde is not None
            else "Not estimable."
        ),
        "allowed_use": "descriptive feasibility/power stress test only",
        "valid_for_h_b1_or_h_b2": False,
    }


def best_case_two_proportion_mde(n_a: int, n_b: int) -> float | None:
    """Approximate two-sided alpha=.05, power=.80 MDE around p=.5.

    This deliberately optimistic calculation treats structures as independent
    and ignores missing predictions, disagreement filtering, and spatial design
    effects.  It is a feasibility diagnostic, not an inferential result.
    """

    if n_a <= 0 or n_b <= 0:
        return None
    z_alpha_plus_power = 1.959963984540054 + 0.8416212335729143
    return float(z_alpha_plus_power * math.sqrt(0.25 / n_a + 0.25 / n_b))


def audit_media(
    frame: pd.DataFrame, dataset_root: Path, resolution_sample: int
) -> dict:
    result: dict[str, dict] = {}
    specs = {
        "street": "street_view_relative_path",
        "remote": "remote_sensing_relative_path",
    }
    for view, column in specs.items():
        if column not in frame.columns:
            result[view] = {"column_present": False}
            continue
        declared = frame[column].astype("string").str.strip()
        has_path = declared.notna() & declared.ne("")
        paths = [
            dataset_root / str(value).replace("\\", os.sep)
            for value in declared[has_path]
        ]
        exists = [path.is_file() for path in paths]
        counts: Counter[str] = Counter()
        failures = 0
        for path in [path for path, present in zip(paths, exists) if present][
            :resolution_sample
        ]:
            try:
                with Image.open(path) as image:
                    counts[f"{image.width}x{image.height} {image.format}"] += 1
            except OSError:
                failures += 1
        result[view] = {
            "column_present": True,
            "manifest_rows": int(len(frame)),
            "declared_paths": len(paths),
            "null_or_blank_path_rows": int((~has_path).sum()),
            "files_present": int(sum(exists)),
            "files_missing": int(len(paths) - sum(exists)),
            "resolution_sample_requested": resolution_sample,
            "resolution_sample_opened": int(sum(counts.values())),
            "resolution_counts": dict(sorted(counts.items())),
            "image_open_failures": failures,
        }
    return result


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")
    if "protocol_role" in frame.columns and (
        _normalized_text(frame["protocol_role"]) == "final_test"
    ).any():
        raise ValueError("final_test predictions are forbidden in this development runner")
    if frame.duplicated(["seed", "pair_id"]).any():
        raise ValueError("Predictions must be unique by (seed, pair_id)")
    pair_ids = frame["pair_id"].astype("string").str.strip()
    if pair_ids.isna().any() or pair_ids.eq("").any():
        raise ValueError("Prediction pair_id cannot be null or blank")
    result = frame.copy()
    seeds = pd.to_numeric(result["seed"], errors="coerce")
    if seeds.isna().any() or not np.equal(seeds, np.floor(seeds)).all():
        raise ValueError("seed must contain integer values")
    result["seed"] = seeds.astype(int)
    for column in ("street_prediction", "remote_prediction"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any():
            raise ValueError(f"{column} must contain finite ordinal values")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integer ordinal values")
        result[column] = values.astype(int)
    blocks = result["spatial_block_id"].astype("string").str.strip()
    if blocks.isna().any() or blocks.eq("").any():
        raise ValueError("spatial_block_id cannot be null or blank")
    result["spatial_block_id"] = blocks
    return result


def validate_component_references(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REFERENCE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Component reference table missing required columns: {sorted(missing)}"
        )
    if frame["pair_id"].duplicated().any():
        raise ValueError("Component reference pair_id must be unique")
    pair_ids = frame["pair_id"].astype("string").str.strip()
    if pair_ids.isna().any() or pair_ids.eq("").any():
        raise ValueError("Component reference pair_id cannot be null or blank")
    semantics = _normalized_text(frame["reference_semantics"])
    if not semantics.eq("damage_dominance").all():
        raise ValueError(
            "reference_semantics must be exactly 'damage_dominance'; construction "
            "or material attributes cannot substitute"
        )
    dominance = _normalized_text(frame["component_dominance"])
    if dominance.isna().any() or dominance.eq("").any():
        raise ValueError("Missing component_dominance must be encoded as 'unknown'")
    invalid = set(dominance.dropna().unique()) - ALLOWED_DOMINANCE
    if invalid:
        raise ValueError(f"Invalid component_dominance values: {sorted(invalid)}")
    provenance = frame["annotation_provenance"].astype("string").str.strip()
    if provenance.isna().any() or provenance.eq("").any():
        raise ValueError("Every component reference needs annotation_provenance")
    result = frame.copy()
    result["component_dominance"] = dominance
    return result


def _cluster_bootstrap_ci(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = statistic(frame)
    blocks = frame["spatial_block_id"].dropna().unique()
    if not len(blocks) or not math.isfinite(point):
        return point, float("nan"), float("nan")
    groups = {block: group for block, group in frame.groupby("spatial_block_id")}
    samples: list[float] = []
    for _ in range(replicates):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        sampled = pd.concat([groups[block] for block in chosen], ignore_index=True)
        value = statistic(sampled)
        if math.isfinite(value):
            samples.append(value)
    if not samples:
        return point, float("nan"), float("nan")
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, float(low), float(high)


def add_component_direction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["disagree"] = result["street_prediction"] != result["remote_prediction"]
    result["signed_direction"] = (
        result["remote_prediction"] - result["street_prediction"]
    )
    expected = result["component_dominance"].map({"roof": 1, "facade": -1})
    result["direction_consistent"] = (
        result["disagree"]
        & expected.notna()
        & (result["signed_direction"] * expected > 0)
    )
    return result


def analyze_component_direction(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    predictions = validate_predictions(predictions)
    references = validate_component_references(references)
    merged = predictions.merge(
        references[["pair_id", "component_dominance"]],
        on="pair_id",
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("Predictions and component references have no pair_id overlap")
    merged = add_component_direction_columns(merged)
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict] = []

    def positive_share(subset: pd.DataFrame) -> float:
        disagreed = subset[subset["disagree"]]
        return (
            float((disagreed["signed_direction"] > 0).mean())
            if len(disagreed)
            else float("nan")
        )

    def contrast(subset: pd.DataFrame) -> float:
        roof = positive_share(subset[subset["component_dominance"] == "roof"])
        facade = positive_share(subset[subset["component_dominance"] == "facade"])
        return roof - facade

    def consistent_share(subset: pd.DataFrame) -> float:
        eligible = subset[
            subset["disagree"]
            & subset["component_dominance"].isin(["roof", "facade"])
        ]
        return (
            float(eligible["direction_consistent"].mean())
            if len(eligible)
            else float("nan")
        )

    for seed, seed_frame in merged.groupby("seed"):
        row: dict[str, int | float] = {
            "seed": int(seed),
            "joined_rows": int(len(seed_frame)),
            "spatial_blocks": int(seed_frame["spatial_block_id"].nunique()),
        }
        for dominance in ("roof", "facade"):
            subset = seed_frame[seed_frame["component_dominance"] == dominance]
            point, low, high = _cluster_bootstrap_ci(
                subset, positive_share, bootstrap_replicates, rng
            )
            row[f"{dominance}_rows"] = int(len(subset))
            row[f"{dominance}_disagreements"] = int(subset["disagree"].sum())
            row[f"{dominance}_positive_direction_share"] = point
            row[f"{dominance}_positive_direction_share_ci_low"] = low
            row[f"{dominance}_positive_direction_share_ci_high"] = high
        for name, statistic in (
            ("h_b1_directional_contrast", contrast),
            ("h_b2_direction_consistent_share", consistent_share),
        ):
            point, low, high = _cluster_bootstrap_ci(
                seed_frame, statistic, bootstrap_replicates, rng
            )
            row[name] = point
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def _artifact_fingerprint(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def discover_eaton_prediction_candidates(search_root: Path) -> list[str]:
    """Inventory filenames that could plausibly be an Eaton prediction export."""

    if not search_root.is_dir():
        return []
    candidates = []
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in {".csv", ".parquet"}:
            continue
        normalized = str(path).casefold()
        if "eaton" in normalized and "prediction" in normalized:
            candidates.append(str(path.resolve()))
    return sorted(candidates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-manifest", type=Path, required=True)
    parser.add_argument("--field-domains", type=Path, required=True)
    parser.add_argument("--whitelist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--component-reference-csv", type=Path)
    parser.add_argument("--prediction-search-root", type=Path, default=Path("outputs"))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--resolution-sample", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap-replicates must be positive")
    if args.resolution_sample < 0:
        raise ValueError("resolution-sample cannot be negative")
    for path in (args.joined_manifest, args.field_domains, args.whitelist):
        if not path.is_file():
            raise FileNotFoundError(path)

    whitelist = load_and_validate_whitelist(args.whitelist)
    # Parsing is part of the input-integrity check even though domain values are
    # not converted into outcomes by this analysis.
    field_domains = json.loads(args.field_domains.read_text(encoding="utf-8"))
    field_service_domains = field_domains.get("field_service", {})
    domain_audit = {
        "published_layer_type_count": len(
            field_service_domains.get("published_layer_types", [])
        ),
        "fields_with_direct_domain": int(
            field_service_domains.get("fields_with_direct_domain", 0)
        ),
        "fields_with_subtype_domain": int(
            field_service_domains.get("fields_with_subtype_domain", 0)
        ),
        "interpretation": (
            "No published component-field domains are available to certify "
            "observed categorical values."
        ),
    }
    joined = pd.read_csv(args.joined_manifest, low_memory=False)
    field_audit, joined_summary = audit_reference_fields(joined, whitelist)
    dataset_root = args.dataset_root or args.joined_manifest.parent.parent
    media_audit = audit_media(joined, dataset_root, args.resolution_sample)
    origin_proxy = audit_restricted_origin_proxy(joined)
    prediction_inventory = discover_eaton_prediction_candidates(
        args.prediction_search_root
    )

    blockers: list[dict[str, str]] = []
    if args.predictions_csv is None:
        blockers.append(
            {
                "code": "missing_eaton_per_view_severity_predictions",
                "detail": (
                    "No genuine Eaton street_prediction/remote_prediction export "
                    "was supplied; predictions are never synthesized from labels. "
                    f"Repository candidate inventory count: {len(prediction_inventory)}."
                ),
            }
        )
    if args.component_reference_csv is None:
        blockers.append(
            {
                "code": "missing_component_damage_dominance_reference",
                "detail": (
                    "The joined DINS material/construction fields cannot be used as "
                    "component-damage dominance labels."
                ),
            }
        )
    if "spatial_block_id" not in joined.columns and args.predictions_csv is None:
        blockers.append(
            {
                "code": "missing_frozen_spatial_dependency_groups",
                "detail": (
                    "The joined manifest has structure IDs but no frozen Eaton "
                    "spatial_block_id; cluster inference cannot be run."
                ),
            }
        )

    statistics: pd.DataFrame | None = None
    validated_predictions: pd.DataFrame | None = None
    validated_references: pd.DataFrame | None = None
    extra_fingerprints: dict[str, dict] = {}
    if args.predictions_csv is not None:
        if not args.predictions_csv.is_file():
            raise FileNotFoundError(args.predictions_csv)
        extra_fingerprints["predictions"] = _artifact_fingerprint(
            args.predictions_csv
        )
        validated_predictions = validate_predictions(pd.read_csv(args.predictions_csv))
    if args.component_reference_csv is not None:
        if not args.component_reference_csv.is_file():
            raise FileNotFoundError(args.component_reference_csv)
        extra_fingerprints["component_reference"] = _artifact_fingerprint(
            args.component_reference_csv
        )
        validated_references = validate_component_references(
            pd.read_csv(args.component_reference_csv)
        )
    if not blockers:
        assert validated_predictions is not None
        assert validated_references is not None
        statistics = analyze_component_direction(
            validated_predictions,
            validated_references,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )

    statistics_path = args.output_dir / "component_direction_statistics.csv"
    if statistics is None and statistics_path.exists():
        raise FileExistsError(
            "Blocked audit refuses to coexist with stale inferential output: "
            f"{statistics_path}. Use a fresh output directory or remove the "
            "previous artifact after verifying its provenance."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    field_audit.to_csv(args.output_dir / "field_availability.csv", index=False)
    if statistics is not None:
        statistics.to_csv(statistics_path, index=False)

    artifact_fingerprints = {
        "joined_manifest": _artifact_fingerprint(args.joined_manifest),
        "field_domains": _artifact_fingerprint(args.field_domains),
        "frozen_whitelist": _artifact_fingerprint(args.whitelist),
        **extra_fingerprints,
    }
    provenance_path = args.joined_manifest.parent / "join_provenance.json"
    if provenance_path.is_file():
        artifact_fingerprints["join_provenance"] = _artifact_fingerprint(
            provenance_path
        )

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "analysis_status": "executed_exploratory" if not blockers else "blocked",
        "claim_scope": (
            "RQ1 Study B development/exploratory only; inspector references are "
            "not image-level visibility labels"
        ),
        "protocol": "docs/results/disagreement_anatomy_protocol.md",
        "artifact_fingerprints": artifact_fingerprints,
        "whitelist": {
            "schema_version": whitelist["schema_version"],
            "frozen_on": whitelist["frozen_on"],
            "field_count": len(whitelist["fields"]),
            "all_component_damage_reference_flags": False,
        },
        "joined_data_audit": joined_summary,
        "field_domain_audit": domain_audit,
        "field_availability_artifact": "field_availability.csv",
        "repository_prediction_inventory": {
            "search_root": str(args.prediction_search_root.resolve()),
            "candidate_count": len(prediction_inventory),
            "candidate_paths": prediction_inventory,
            "rule": "path contains both 'eaton' and 'prediction' and is CSV/Parquet",
        },
        "media_audit": media_audit,
        "restricted_fire_origin_proxy": origin_proxy,
        "main_hypothesis_power": {
            "estimable": statistics is not None,
            "reason_if_not_estimable": (
                "There are zero valid component-damage dominance references and "
                "zero supplied per-view Eaton prediction rows; a main-effect MDE "
                "or disagreement-conditioned H-B2 precision cannot be estimated."
                if statistics is None
                else None
            ),
            "h_b2_independent_disagreements_needed_for_approx_95pct_half_width": {
                "0.10": 97,
                "0.05": 385,
                "caveat": "before spatial design effect and stratum imbalance",
            },
        },
        "prerequisites": {
            "genuine_eaton_per_view_severity_predictions_supplied": (
                args.predictions_csv is not None
            ),
            "provenance_bearing_component_damage_dominance_supplied": (
                args.component_reference_csv is not None
            ),
            "component_predictions_synthesized": False,
            "construction_fields_relabelled_as_damage": False,
            "stale_inferential_output_guard": True,
        },
        "blockers": blockers,
        "inference_outputs": (
            ["component_direction_statistics.csv"] if statistics is not None else []
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis_status": audit["analysis_status"],
                "blocker_codes": [item["code"] for item in blockers],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
