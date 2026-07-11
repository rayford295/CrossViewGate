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

from crossview_conflict.data.dins import (  # noqa: E402
    DEFAULT_BUSINESS_KEY_CANDIDATES,
    DEFAULT_EATON_WHERE,
    EATON_DINS_FIELDS_SERVICE_URL,
    EATON_DINS_PUBLIC_SERVICE_URL,
    artifact_paths,
    build_join_provenance,
    compute_field_statistics,
    extract_field_domains,
    features_to_frame,
    fetch_dins_layer,
    fetch_layer_metadata,
    infer_metadata_from_geojson,
    join_dins_fields,
    load_geojson,
    load_service_metadata,
    service_last_edit_utc,
    write_dins_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch CAL FIRE Eaton DINS fields and auditably join them to an Eaton "
            "manifest. The default is a dry run: it may read the official service "
            "but writes nothing until --write is supplied."
        )
    )
    parser.add_argument("--manifest-csv", required=True, help="Existing Eaton manifest CSV.")
    parser.add_argument(
        "--service-url",
        default=EATON_DINS_FIELDS_SERVICE_URL,
        help="Official ArcGIS field layer URL.",
    )
    parser.add_argument(
        "--where",
        default=DEFAULT_EATON_WHERE,
        help="ArcGIS SQL filter used for both count and paginated feature queries.",
    )
    parser.add_argument(
        "--public-reference-service-url",
        default=EATON_DINS_PUBLIC_SERVICE_URL,
        help=(
            "Public attachment-layer URL whose metadata/domains are recorded for "
            "cross-service provenance; its features are not joined."
        ),
    )
    parser.add_argument(
        "--no-public-reference-metadata",
        action="store_true",
        help="Do not fetch public attachment-layer reference metadata in network mode.",
    )
    parser.add_argument(
        "--source-geojson",
        help="Use a local field-layer GeoJSON snapshot and make no network requests.",
    )
    parser.add_argument(
        "--source-metadata",
        help="Local ArcGIS metadata JSON paired with --source-geojson.",
    )
    parser.add_argument(
        "--reference-metadata",
        help="Optional local public attachment-layer metadata JSON for offline mode.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dins/eaton_field_join",
        help="Artifact directory used only with --write.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the enriched copy and provenance artifacts. Without this flag: dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing generated artifacts (requires --write).",
    )
    parser.add_argument("--page-size", type=int, help="ArcGIS result page size.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=10.0,
        help="Maximum nearest-point distance when no unique stable key matches.",
    )
    parser.add_argument(
        "--ambiguity-margin-m",
        type=float,
        default=1.0,
        help=(
            "Do not enrich when the two nearest candidates differ by at most this many metres."
        ),
    )
    parser.add_argument(
        "--business-key",
        action="append",
        metavar="FIELD[,FIELD...]",
        help=(
            "Stable key candidate, repeatable in priority order. OBJECTID/OID/FID are rejected. "
            "Defaults to GLOBALID, incident+APN+structure type, APN+structure type, then "
            "site address+structure type."
        ),
    )
    parser.add_argument("--manifest-latitude-column", default="latitude")
    parser.add_argument("--manifest-longitude-column", default="longitude")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _business_keys(specifications: list[str] | None) -> tuple[tuple[str, ...], ...]:
    if not specifications:
        return DEFAULT_BUSINESS_KEY_CANDIDATES
    parsed: list[tuple[str, ...]] = []
    for specification in specifications:
        fields = tuple(field.strip() for field in specification.split(",") if field.strip())
        if not fields:
            raise ValueError(f"Empty --business-key specification: {specification!r}")
        parsed.append(fields)
    return tuple(parsed)


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
]:
    if args.source_geojson:
        geojson = load_geojson(args.source_geojson)
        metadata = (
            load_service_metadata(args.source_metadata)
            if args.source_metadata
            else infer_metadata_from_geojson(geojson)
        )
        reference_metadata = (
            load_service_metadata(args.reference_metadata)
            if args.reference_metadata
            else None
        )
        properties = geojson.get("properties") or {}
        fetch_provenance: dict[str, object] = {
            "source_mode": "local_geojson",
            "service_url": properties.get("source_service_url", args.service_url),
            "query_where": properties.get("query_where"),
            "retrieved_at_utc": properties.get("retrieved_at_utc"),
            "service_last_edit_utc": service_last_edit_utc(metadata),
            "record_count_fetched": len(geojson.get("features", [])),
            "pagination": "already materialized local snapshot",
            "network_requests_made": False,
        }
        return metadata, geojson, fetch_provenance, reference_metadata

    if args.source_metadata or args.reference_metadata:
        raise ValueError(
            "--source-metadata/--reference-metadata require --source-geojson for offline mode"
        )
    metadata, geojson, fetch_provenance = fetch_dins_layer(
        args.service_url,
        where=args.where,
        page_size=args.page_size,
        timeout=args.timeout,
    )
    reference_metadata = None
    if not args.no_public_reference_metadata:
        reference_metadata = fetch_layer_metadata(
            args.public_reference_service_url,
            timeout=args.timeout,
        )
    return metadata, geojson, fetch_provenance, reference_metadata


def main() -> None:
    args = parse_args()
    if args.overwrite and not args.write:
        raise ValueError("--overwrite has no effect in dry-run mode; add --write explicitly")

    manifest_path = Path(args.manifest_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    metadata, geojson, fetch_provenance, reference_metadata = _load_inputs(args)
    source = features_to_frame(geojson)
    field_service_url = str(fetch_provenance.get("service_url") or args.service_url)
    enriched, join_summary = join_dins_fields(
        manifest,
        source,
        business_key_candidates=_business_keys(args.business_key),
        manifest_latitude_column=args.manifest_latitude_column,
        manifest_longitude_column=args.manifest_longitude_column,
        tolerance_m=args.tolerance_m,
        ambiguity_margin_m=args.ambiguity_margin_m,
        service_url=field_service_url,
        service_last_edit=service_last_edit_utc(metadata),
    )

    statistics = compute_field_statistics(source, metadata)
    domains: dict[str, object] = {
        "generated_at_utc": _utc_now(),
        "field_service": extract_field_domains(
            metadata, service_url=field_service_url
        ),
    }
    if reference_metadata is not None:
        domains["public_attachment_reference_service"] = extract_field_domains(
            reference_metadata,
            service_url=args.public_reference_service_url,
        )
    service_metadata_artifact: dict[str, object] = {
        "retrieved_or_loaded_at_utc": _utc_now(),
        "field_service_url": field_service_url,
        "field_service_metadata": metadata,
        "public_attachment_reference_service_url": (
            args.public_reference_service_url if reference_metadata is not None else None
        ),
        "public_attachment_reference_metadata": reference_metadata,
        "cross_service_objectid_policy": (
            "The public attachment layer and expanded field layer are distinct services; "
            "their OBJECTID values are never equated."
        ),
    }
    provenance = build_join_provenance(
        fetch_provenance=fetch_provenance,
        join_summary=join_summary,
        manifest_path=manifest_path,
        source_geojson=geojson,
        source_geojson_path=args.source_geojson,
    )

    planned_paths = artifact_paths(args.output_dir)
    written_paths: dict[str, Path] | None = None
    if args.write:
        written_paths = write_dins_artifacts(
            args.output_dir,
            geojson=geojson,
            service_metadata_artifact=service_metadata_artifact,
            domains=domains,
            statistics=statistics,
            joined_manifest=enriched,
            provenance=provenance,
            overwrite=args.overwrite,
        )

    top_null = sorted(
        statistics["fields"],
        key=lambda field: field["null_or_blank_rate"] or 0.0,
        reverse=True,
    )[:5]
    top_unknown = sorted(
        statistics["fields"],
        key=lambda field: field["unknown_rate_all_rows"] or 0.0,
        reverse=True,
    )[:5]
    report = {
        "mode": "write" if args.write else "dry_run_no_files_written",
        "network_mode": not bool(args.source_geojson),
        "fetch": fetch_provenance,
        "join": {
            key: value
            for key, value in join_summary.items()
            if key != "source_field_output_columns"
        },
        "field_statistics": {
            "field_count": len(statistics["fields"]),
            "top_null_or_blank": top_null,
            "top_unknown": top_unknown,
        },
        "artifacts": {
            key: str(path.resolve())
            for key, path in (written_paths or planned_paths).items()
        },
        "write_required": not args.write,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
