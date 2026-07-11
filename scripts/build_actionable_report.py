"""Build tile priorities, a geodesic inspection route, and an action report.

The command accepts the P0.7 ``per_sample_decision.csv`` artifact and retains a
compatibility path for P0.2 per-sample evidence.  Route distances are Haversine
straight-line proxies, not road-network distances or travel-time estimates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crossview_conflict.decision.routing import (
    DEFAULT_RANDOM_SEED,
    POLICIES,
    ROUTING_SCHEMA_VERSION,
    RouteResult,
    RoutingInputInfo,
    annotate_reference_route_ranks,
    build_tile_priority,
    dumps_geojson,
    evaluate_route,
    metrics_frame,
    normalize_routing_input,
    simulate_route,
)


DEFAULT_MAX_STOPS = 20


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-sample CrossViewGate decisions/evidence into auditable tile "
            "priorities, a Haversine route proxy, and an actionable Markdown report."
        )
    )
    parser.add_argument(
        "--input-csv",
        "--per-sample-csv",
        "--decisions-csv",
        dest="input_csv",
        required=True,
        help="P0.7 per_sample_decision.csv or compatible P0.2 evidence CSV.",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/actionable_report")
    parser.add_argument("--policy", choices=POLICIES, default="joint")
    parser.add_argument(
        "--max-stops",
        "--k-stops",
        type=int,
        default=None,
        help=(
            "Maximum inspection stops. If neither this nor a distance budget is supplied, "
            f"the default is {DEFAULT_MAX_STOPS} stops."
        ),
    )
    parser.add_argument(
        "--distance-budget-km",
        "--travel-budget-km",
        type=float,
        default=None,
        help="Maximum cumulative Haversine travel distance in kilometers.",
    )
    parser.add_argument("--start-lat", type=float, default=None)
    parser.add_argument("--start-lon", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--severe-labels",
        default="",
        help=(
            "Comma-separated target labels counted as severe. When omitted and target exists, "
            "the highest ordinal/numeric label is inferred and disclosed."
        ),
    )
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--lat-col", default="auto")
    parser.add_argument("--lon-col", default="auto")
    parser.add_argument("--tile-col", default="auto")
    parser.add_argument("--risk-score-col", default="auto")
    parser.add_argument("--disposition-col", default="auto")
    parser.add_argument("--severity-probability-col", default="auto")
    parser.add_argument("--hard-conflict-col", default="auto")
    parser.add_argument("--js-divergence-col", default="auto")
    return parser.parse_args(argv)


def _parse_severe_labels(value: str) -> tuple[str, ...] | None:
    labels = tuple(part.strip() for part in value.split(",") if part.strip())
    return labels or None


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        rendered = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def render_actionable_report(
    *,
    input_path: Path,
    info: RoutingInputInfo,
    tile_priority: pd.DataFrame,
    selected_route: RouteResult,
    comparison: pd.DataFrame,
) -> str:
    """Render a deterministic, human-readable report from the routing artifacts."""

    selected_metrics = comparison.loc[comparison["policy"] == selected_route.policy].iloc[0]
    labeled_samples = int(tile_priority["labeled_sample_count"].sum())
    total_samples = int(tile_priority["sample_count"].sum())
    budget_description: list[str] = []
    if selected_route.max_stops is not None:
        budget_description.append(f"at most {selected_route.max_stops} stops")
    if selected_route.distance_budget_km is not None:
        budget_description.append(
            f"{selected_route.distance_budget_km:.3f} km cumulative travel"
        )
    if not budget_description:
        budget_description.append("all reachable tiles")

    lines = [
        "# CrossViewGate actionable inspection report",
        "",
        "## Operational scope",
        "",
        f"- Input: `{input_path}`",
        f"- Evidence: {total_samples} samples aggregated into {len(tile_priority)} tiles.",
        f"- Selected policy: `{selected_route.policy}` under {' and '.join(budget_description)}.",
        (
            f"- Origin: ({selected_route.start_latitude:.6f}, "
            f"{selected_route.start_longitude:.6f}); the first leg is included in the budget."
        ),
        (
            "- Distance model: WGS84 Haversine/geodesic straight-line proxy. "
            "This is **not road-network routing**, does not model barriers or access, "
            "and is not a travel-time estimate."
        ),
        "- The route is open: no return-to-origin leg is included.",
    ]
    if info.labels_available:
        lines.append(
            "- Label mode: descriptive evaluation enabled for "
            f"{labeled_samples}/{total_samples} samples; severe labels = "
            f"`{', '.join(map(str, info.severe_labels))}`. Targets do not influence routing."
        )
    else:
        lines.append(
            "- Label mode: no target labels supplied. Routing and all operational artifacts "
            "remain available; severe-discovery metrics are reported as `n/a`."
        )

    lines.extend(
        [
            "",
            "## Selected route summary",
            "",
            *_markdown_table(
                [
                    "policy",
                    "stops",
                    "distance km",
                    "tile coverage",
                    "sample coverage",
                    "severe discoveries",
                    "severe Recall@budget",
                    "NDCG@budget",
                ],
                [
                    [
                        selected_metrics["policy"],
                        int(selected_metrics["stop_count"]),
                        _format_number(selected_metrics["distance_km"]),
                        _format_number(selected_metrics["tile_coverage"]),
                        _format_number(selected_metrics["sample_coverage"]),
                        _format_number(selected_metrics["severe_discoveries"]),
                        _format_number(selected_metrics["severe_recall_at_budget"]),
                        _format_number(selected_metrics["ndcg_at_budget"]),
                    ]
                ],
            ),
            "",
            "## Inspection queue",
            "",
        ]
    )
    if selected_route.stops.empty:
        lines.append("No tile fits the declared resource budget.")
    else:
        queue_rows = []
        for _, stop in selected_route.stops.iterrows():
            queue_rows.append(
                [
                    int(stop["stop_order"]),
                    stop["tile_id"],
                    f"{float(stop['latitude']):.6f}",
                    f"{float(stop['longitude']):.6f}",
                    _format_number(stop["priority_score"]),
                    _format_number(stop["severity_probability"]),
                    _format_number(stop["uncertainty_score"]),
                    _format_number(stop["conflict_score"]),
                    stop["disposition"],
                    _format_number(stop["leg_distance_km"]),
                    _format_number(stop["cumulative_distance_km"]),
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "stop",
                    "tile",
                    "lat",
                    "lon",
                    "priority",
                    "severity p",
                    "uncertainty",
                    "conflict",
                    "disposition",
                    "leg km",
                    "cumulative km",
                ],
                queue_rows,
            )
        )

    lines.extend(["", "## Baseline comparison", ""])
    comparison_rows = []
    for _, row in comparison.iterrows():
        comparison_rows.append(
            [
                row["policy"],
                int(row["stop_count"]),
                _format_number(row["distance_km"]),
                _format_number(row["tile_coverage"]),
                _format_number(row["severe_discoveries"]),
                _format_number(row["severe_recall_at_budget"]),
                _format_number(row["ndcg_at_budget"]),
                _format_number(row["first_severe_distance_km"]),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "policy",
                "stops",
                "distance km",
                "tile coverage",
                "severe discoveries",
                "Recall@budget",
                "NDCG@budget",
                "first severe km",
            ],
            comparison_rows,
        )
    )

    lines.extend(
        [
            "",
            "The random baseline uses the fixed seed "
            f"`{selected_route.random_seed}`. All policies use the same origin and resource limits.",
            "",
            "## Priority definitions and audit trail",
            "",
            "Tile components are maximum sample-level values within each tile; tile coordinates "
            "are the mean sample coordinates. Scores are:",
            "",
            "- `severity_only = severity_probability`",
            "- `uncertainty_only = normalized risk_score`",
            "- `conflict_only = max(hard_conflict, clipped JS / ln(2))`",
            "- `reliability_aware = 0.50 severity + 0.30 uncertainty + 0.20 escalation`",
            "- `joint = 0.40 severity + 0.25 uncertainty + 0.25 conflict + 0.10 escalation`",
            "- `nearest_neighbor` has no evidence score; it greedily minimizes the next leg.",
            "",
            "`tile_priority.csv` records every component, policy score, global rank, reference "
            "route rank, and selected-route leg/cumulative distance. `inspection_route.geojson` "
            "stores point coordinates in GeoJSON order `[longitude, latitude]` plus an origin-to-stop "
            "LineString.",
            "",
            "### Resolved fields",
            "",
            *_markdown_table(
                ["field", "source"],
                [
                    ["latitude", info.latitude_source],
                    ["longitude", info.longitude_source],
                    ["tile id", info.tile_source],
                    ["severity probability", info.severity_probability_source],
                    ["risk score", info.risk_score_source],
                    ["disposition", info.disposition_source],
                    ["conflict", info.conflict_source],
                    ["target", info.target_source or "not supplied"],
                ],
            ),
        ]
    )
    if info.warnings:
        lines.extend(["", "### Compatibility notes", ""])
        lines.extend(f"- {warning}" for warning in info.warnings)
    lines.extend(
        [
            "",
            "## Metric interpretation",
            "",
            "Severe discoveries count labeled severe samples reached by the route. "
            "Recall@budget divides those discoveries by all labeled severe samples. "
            "NDCG@budget uses per-tile severe-sample count as graded relevance at the actual "
            "number of stops. These are descriptive metrics and must not be used to tune a "
            "policy on the final test event.",
            "",
        ]
    )
    return "\n".join(lines)


def _attach_selected_route_audit(
    tile_priority: pd.DataFrame, selected_route: RouteResult
) -> pd.DataFrame:
    output = tile_priority.copy()
    audit_columns = (
        "tile_id",
        "stop_order",
        "priority_score",
        "leg_distance_km",
        "cumulative_distance_km",
        "remaining_distance_budget_km",
    )
    if selected_route.stops.empty:
        for column in audit_columns[1:]:
            output[f"selected_{column}"] = pd.NA
        return output
    audit = selected_route.stops[list(audit_columns)].rename(
        columns={column: f"selected_{column}" for column in audit_columns if column != "tile_id"}
    )
    return output.merge(audit, on="tile_id", how="left", validate="one_to_one")


def build_outputs(
    frame: pd.DataFrame,
    *,
    input_path: Path,
    output_dir: Path,
    policy: str = "joint",
    max_stops: int | None = None,
    distance_budget_km: float | None = None,
    start_latitude: float | None = None,
    start_longitude: float | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    severe_labels: Sequence[Any] | None = None,
    target_col: str = "target",
    latitude_col: str = "auto",
    longitude_col: str = "auto",
    tile_col: str = "auto",
    risk_score_col: str = "auto",
    disposition_col: str = "auto",
    severity_probability_col: str = "auto",
    hard_conflict_col: str = "auto",
    js_divergence_col: str = "auto",
) -> dict[str, Any]:
    """Build and write the three routing-MVP artifacts."""

    normalized, info = normalize_routing_input(
        frame,
        latitude_col=latitude_col,
        longitude_col=longitude_col,
        tile_col=tile_col,
        risk_score_col=risk_score_col,
        disposition_col=disposition_col,
        severity_probability_col=severity_probability_col,
        hard_conflict_col=hard_conflict_col,
        js_divergence_col=js_divergence_col,
        target_col=target_col,
        severe_labels=severe_labels,
    )
    tiles = build_tile_priority(normalized, random_seed=random_seed)
    if max_stops is None and distance_budget_km is None:
        max_stops = DEFAULT_MAX_STOPS
    tiles = annotate_reference_route_ranks(
        tiles,
        start_latitude=start_latitude,
        start_longitude=start_longitude,
        random_seed=random_seed,
    )

    routes: dict[str, RouteResult] = {}
    metric_rows: list[dict[str, Any]] = []
    for baseline in POLICIES:
        route = simulate_route(
            tiles,
            policy=baseline,
            max_stops=max_stops,
            distance_budget_km=distance_budget_km,
            start_latitude=start_latitude,
            start_longitude=start_longitude,
            random_seed=random_seed,
        )
        routes[baseline] = route
        metric_rows.append(evaluate_route(route, tiles))
    selected_route = routes[policy]
    comparison = metrics_frame(metric_rows)
    audited_tiles = _attach_selected_route_audit(tiles, selected_route)
    audited_tiles.insert(0, "schema_version", ROUTING_SCHEMA_VERSION)
    audited_tiles.insert(1, "selected_policy", policy)
    score_column = f"priority_{policy}"
    audited_tiles.insert(
        2,
        "selected_policy_score",
        audited_tiles[score_column] if score_column in audited_tiles else float("nan"),
    )
    audited_tiles.insert(
        3, "selected_policy_rank", audited_tiles[f"route_rank_{policy}"].astype(int)
    )
    audited_tiles = audited_tiles.sort_values(
        ["selected_policy_rank", "tile_id"], kind="mergesort"
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_path = output_dir / "tile_priority.csv"
    route_path = output_dir / "inspection_route.geojson"
    report_path = output_dir / "actionable_report.md"
    audited_tiles.to_csv(tile_path, index=False)
    route_path.write_text(dumps_geojson(selected_route), encoding="utf-8")
    report_path.write_text(
        render_actionable_report(
            input_path=input_path,
            info=info,
            tile_priority=audited_tiles,
            selected_route=selected_route,
            comparison=comparison,
        ),
        encoding="utf-8",
    )

    selected_metrics = comparison.loc[comparison["policy"] == policy].iloc[0].to_dict()
    return {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "selected_policy": policy,
        "labels_available": info.labels_available,
        "severe_labels": list(info.severe_labels),
        "selected_route": {
            key: (None if pd.isna(value) else value) for key, value in selected_metrics.items()
        },
        "distance_model": "haversine_geodesic_straight_line_proxy",
        "road_network_routing": False,
        "artifacts": {
            "tile_priority": tile_path.name,
            "inspection_route": route_path.name,
            "actionable_report": report_path.name,
        },
        "input_info": info.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if (args.start_lat is None) != (args.start_lon is None):
        raise ValueError("--start-lat and --start-lon must be supplied together")
    input_path = Path(args.input_csv)
    frame = pd.read_csv(input_path)
    summary = build_outputs(
        frame,
        input_path=input_path,
        output_dir=Path(args.output_dir),
        policy=args.policy,
        max_stops=args.max_stops,
        distance_budget_km=args.distance_budget_km,
        start_latitude=args.start_lat,
        start_longitude=args.start_lon,
        random_seed=args.random_seed,
        severe_labels=_parse_severe_labels(args.severe_labels),
        target_col=args.target_col,
        latitude_col=args.lat_col,
        longitude_col=args.lon_col,
        tile_col=args.tile_col,
        risk_score_col=args.risk_score_col,
        disposition_col=args.disposition_col,
        severity_probability_col=args.severity_probability_col,
        hard_conflict_col=args.hard_conflict_col,
        js_divergence_col=args.js_divergence_col,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
