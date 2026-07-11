from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SCHEMA = "cvian-active-view-development-evaluation-v2"
SUMMARY_SCHEMA = "cvian-active-view-development-summary-v2"
EXPECTED_POLICIES = (
    "clockwise",
    "random",
    "farthest",
    "learned",
    "max_building_privileged",
    "max_confidence_privileged",
    "entropy_reduction_privileged",
    "oracle_cost",
)
EXPECTED_BUDGETS = tuple(range(1, 9))
EXPECTED_TEST_ROWS = 415
EXPECTED_SEEDS = (42, 123, 456, 789, 1011)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the five-seed CVIAN active-view development evaluation."
    )
    parser.add_argument(
        "--experiment-root",
        default="outputs/analysis/active_view_cvian_spatial_v1/experiment",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-doc",
        default="docs/results/active_view_cvian_spatial_v1.md",
    )
    return parser.parse_args()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fingerprint_payload(record: dict[str, object]) -> dict[str, object]:
    required = (
        "schema_version",
        "config",
        "source_sha256",
        "input_sha256",
        "runtime_versions",
        "input_semantics",
    )
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Completion lacks fingerprint field {missing[0]!r}")
    return {field: record[field] for field in required}


def _format(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _sequence_connected_components(decisions: pd.DataFrame) -> dict[str, str]:
    pairs = decisions[["spatial_block_id", "sequence_id"]].drop_duplicates().copy()
    pairs["spatial_block_id"] = pairs["spatial_block_id"].astype(str)
    pairs["sequence_id"] = pairs["sequence_id"].astype(str)
    blocks = sorted(pairs["spatial_block_id"].unique())
    parent = {block: block for block in blocks}

    def find(block: str) -> str:
        while parent[block] != block:
            parent[block] = parent[parent[block]]
            block = parent[block]
        return block

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for _, group in pairs.groupby("sequence_id"):
        linked = sorted(group["spatial_block_id"].unique())
        for block in linked[1:]:
            union(linked[0], block)
    roots = sorted({find(block) for block in blocks})
    names = {
        root: f"dependency_component_{index:02d}"
        for index, root in enumerate(roots)
    }
    return {block: names[find(block)] for block in blocks}


def _validate_seed_tables(
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    seed: int,
    view_cost: float,
) -> None:
    expected_pairs = {
        (policy, budget)
        for policy in EXPECTED_POLICIES
        for budget in EXPECTED_BUDGETS
    }
    if metrics.duplicated(["policy", "view_count"]).any():
        raise ValueError(f"Duplicate metric key for seed {seed}")
    metric_pairs = set(
        zip(metrics["policy"].astype(str), metrics["view_count"].astype(int))
    )
    if metric_pairs != expected_pairs or len(metrics) != len(expected_pairs):
        raise ValueError(f"Incomplete policy/budget metric grid for seed {seed}")
    decision_key = ["policy", "view_count", "sample_id"]
    if decisions.duplicated(decision_key).any():
        raise ValueError(f"Duplicate per-sample decision key for seed {seed}")
    if decisions["sample_id"].nunique() != EXPECTED_TEST_ROWS:
        raise ValueError(f"Unexpected development-test sample count for seed {seed}")
    expected_decisions = len(expected_pairs) * EXPECTED_TEST_ROWS
    if len(decisions) != expected_decisions:
        raise ValueError(f"Incomplete decision table for seed {seed}")
    group_sizes = decisions.groupby(["policy", "view_count"]).size()
    if set(group_sizes.index) != expected_pairs or not (
        group_sizes.to_numpy() == EXPECTED_TEST_ROWS
    ).all():
        raise ValueError(f"Policy/budget sample coverage mismatch for seed {seed}")
    metadata_columns = ["target", "spatial_block_id", "sequence_id"]
    if (
        decisions.groupby("sample_id")[metadata_columns]
        .nunique(dropna=False)
        .to_numpy()
        > 1
    ).any():
        raise ValueError(f"Per-sample metadata changes across decisions for seed {seed}")

    for row in decisions.itertuples(index=False):
        revealed = [int(value) for value in str(row.revealed_sectors).split(",")]
        if (
            len(revealed) != int(row.view_count)
            or len(set(revealed)) != len(revealed)
            or 0 not in revealed
            or int(row.last_sector) not in revealed
            or any(value not in range(8) for value in revealed)
        ):
            raise ValueError(
                f"Invalid reveal trajectory for seed {seed}, sample {row.sample_id}"
            )

    metric_lookup = metrics.set_index(["policy", "view_count"])
    for pair, group in decisions.groupby(["policy", "view_count"], sort=False):
        target = group["target"].to_numpy(dtype=np.int64)
        prediction = group["prediction"].to_numpy(dtype=np.int64)
        probabilities = group[["prob_0", "prob_1", "prob_2"]].to_numpy(dtype=float)
        severe = target == 2
        operational = group["operational_cost"].to_numpy(dtype=float)
        budget = int(pair[1])
        acquisition = (budget - 1) * view_cost
        nll = -np.log(
            np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0)
        ).mean()
        clipped = np.clip(probabilities, 1e-12, 1.0)
        entropy = -(clipped * np.log(clipped)).sum(axis=1).mean()
        actual = {
            "accuracy": accuracy_score(target, prediction),
            "macro_f1": f1_score(target, prediction, average="macro"),
            "severe_recall": recall_score(
                severe, prediction == 2, zero_division=0
            ),
            "severe_miss_rate": ((prediction != 2) & severe).sum()
            / max(severe.sum(), 1),
            "extreme_error_rate": (np.abs(prediction - target) == 2).mean(),
            "operational_cost": operational.mean(),
            "normalized_operational_cost": operational.mean() / 8.0,
            "nll": nll,
            "mean_entropy": entropy,
            "additional_acquisition_cost": acquisition,
            "total_cost": operational.mean() + acquisition,
            "mean_cumulative_action_regret": group[
                "cumulative_action_regret"
            ].mean(),
        }
        reported = metric_lookup.loc[pair]
        for field, expected in actual.items():
            if not np.isclose(float(reported[field]), float(expected), atol=1e-9):
                raise ValueError(
                    f"Metric {field} disagrees with decisions for seed {seed}, {pair}"
                )


def _dependency_cluster_bootstrap_comparison(
    decisions: pd.DataFrame,
    *,
    baseline: str,
    budget: int,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    subset = decisions[
        (decisions["view_count"] == budget)
        & decisions["policy"].isin(["learned", baseline])
    ].copy()
    subset["severe_miss"] = (
        (subset["target"] == 2) & (subset["prediction"] != 2)
    ).astype(float)
    pivot = subset.pivot_table(
        index=["seed", "sample_id", "spatial_block_id", "sequence_id", "target"],
        columns="policy",
        values=["operational_cost", "severe_miss"],
        aggfunc="first",
    )
    if pivot.isna().any().any():
        raise ValueError(f"Incomplete paired decisions for {baseline}")
    rows = pivot.reset_index()
    rows["cost_improvement"] = (
        rows[("operational_cost", baseline)]
        - rows[("operational_cost", "learned")]
    )
    rows["severe_miss_improvement"] = (
        rows[("severe_miss", baseline)] - rows[("severe_miss", "learned")]
    )
    flat = pd.DataFrame(
        {
            "seed": rows[("seed", "")].astype(int),
            "block": rows[("spatial_block_id", "")].astype(str),
            "sequence": rows[("sequence_id", "")].astype(str),
            "target": rows[("target", "")].astype(int),
            "cost_improvement": rows["cost_improvement"].to_numpy(),
            "severe_miss_improvement": rows["severe_miss_improvement"].to_numpy(),
        }
    )
    cost_by_seed = flat.groupby("seed")["cost_improvement"].mean()
    severe_rows = flat[flat["target"] == 2]
    severe_by_seed = severe_rows.groupby("seed")["severe_miss_improvement"].mean()
    component_map = _sequence_connected_components(
        flat.rename(
            columns={"block": "spatial_block_id", "sequence": "sequence_id"}
        )
    )
    flat["dependency_component"] = flat["block"].map(component_map)
    severe_rows = flat[flat["target"] == 2]
    component_cost = (
        flat.groupby(["seed", "dependency_component"])["cost_improvement"]
        .mean()
        .unstack(0)
    )
    component_severe = (
        severe_rows.groupby(["seed", "dependency_component"])["severe_miss_improvement"]
        .mean()
        .unstack(0)
    )
    blocks = sorted(flat["block"].unique())
    components = sorted(flat["dependency_component"].unique())
    rng = np.random.default_rng(seed)
    cost_samples = np.empty(resamples, dtype=np.float64)
    severe_samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.choice(components, size=len(components), replace=True)
        cost_samples[index] = float(component_cost.loc[sampled].to_numpy().mean())
        available = [
            component for component in sampled if component in component_severe.index
        ]
        severe_samples[index] = (
            float(component_severe.loc[available].to_numpy().mean())
            if available
            else np.nan
        )
    severe_samples = severe_samples[np.isfinite(severe_samples)]
    return {
        "baseline": baseline,
        "budget": budget,
        "test_spatial_blocks": len(blocks),
        "sequence_connected_dependency_components": len(components),
        "sample_weighted_cost_improvement": float(flat["cost_improvement"].mean()),
        "dependency_component_macro_cost_improvement": float(
            component_cost.to_numpy().mean()
        ),
        "dependency_component_bootstrap_cost_ci95": [
            float(np.quantile(cost_samples, 0.025)),
            float(np.quantile(cost_samples, 0.975)),
        ],
        "sample_weighted_severe_miss_improvement": float(
            severe_rows["severe_miss_improvement"].mean()
        ),
        "dependency_component_macro_severe_miss_improvement": float(
            component_severe.to_numpy().mean()
        ),
        "dependency_component_bootstrap_severe_miss_ci95": [
            float(np.quantile(severe_samples, 0.025)),
            float(np.quantile(severe_samples, 0.975)),
        ],
        "cost_improvement_positive_seeds": int((cost_by_seed > 0).sum()),
        "severe_miss_improvement_positive_seeds": int((severe_by_seed > 0).sum()),
        "seed_count": int(len(cost_by_seed)),
    }


def _markdown_table(aggregate: pd.DataFrame, budget: int) -> list[str]:
    rows = aggregate[aggregate["view_count"] == budget].sort_values("operational_cost_mean")
    lines = [
        "| Policy | Scope | Macro-F1 | Severe recall | Operational cost | Total cost (view cost 0.5) | Regret |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        scope = "online" if row.policy in {"clockwise", "random", "farthest", "learned"} else "privileged"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.policy}`",
                    scope,
                    _format(row.macro_f1_mean, row.macro_f1_std),
                    _format(row.severe_recall_mean, row.severe_recall_std),
                    _format(row.operational_cost_mean, row.operational_cost_std),
                    _format(row.total_cost_mean, row.total_cost_std),
                    _format(
                        row.mean_cumulative_action_regret_mean,
                        row.mean_cumulative_action_regret_std,
                    ),
                ]
            )
            + " |"
        )
    return lines


def main() -> None:
    args = parse_args()
    experiment_root = _resolve(args.experiment_root)
    output_doc = _resolve(args.output_doc)
    metrics_frames = []
    decision_frames = []
    completion = []
    runner_path = REPO_ROOT / "scripts" / "run_cvian_active_view_experiment.py"
    runner_sha256 = _sha256(runner_path)
    active_view_sha256 = _sha256(
        REPO_ROOT / "crossview_conflict" / "decision" / "active_view.py"
    )
    for seed_dir in sorted(experiment_root.glob("seed*")):
        completion_path = seed_dir / "evaluation_complete.json"
        if not completion_path.is_file():
            raise FileNotFoundError(f"Missing completion attestation: {completion_path}")
        record = json.loads(completion_path.read_text(encoding="utf-8"))
        if record.get("schema_version") != EVALUATION_SCHEMA:
            raise ValueError(f"Stale completion schema: {completion_path}")
        source_sha256 = record.get("source_sha256", {})
        if not isinstance(source_sha256, dict) or source_sha256.get("runner") != runner_sha256:
            raise ValueError(f"Runner hash changed after evaluation: {completion_path}")
        if source_sha256.get("active_view") != active_view_sha256:
            raise ValueError(f"Active-view implementation changed: {completion_path}")
        if _canonical_hash(_fingerprint_payload(record)) != record.get("run_fingerprint"):
            raise ValueError(f"Invalid run fingerprint: {completion_path}")
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"Missing fingerprinted config: {completion_path}")
        for field in ("main_budget", "view_cost", "cost_matrix", "evaluation_status"):
            if record.get(field) != config.get(field):
                raise ValueError(
                    f"Top-level {field} disagrees with fingerprinted config: {completion_path}"
                )
        for filename, field in (
            ("metrics.csv", "metrics_sha256"),
            ("per_sample_decisions.csv", "decisions_sha256"),
            ("policy_artifact.pt", "artifact_sha256"),
            ("training_history.json", "training_history_sha256"),
        ):
            artifact = seed_dir / filename
            if not artifact.is_file() or _sha256(artifact) != record.get(field):
                raise ValueError(f"Artifact hash mismatch: {artifact}")
        seed = int(seed_dir.name.removeprefix("seed"))
        if int(record.get("seed", -1)) != seed or int(config["seed"]) != seed:
            raise ValueError(f"Seed identity mismatch: {completion_path}")
        seed_metrics = pd.read_csv(seed_dir / "metrics.csv")
        seed_decisions = pd.read_csv(seed_dir / "per_sample_decisions.csv")
        if set(seed_metrics["seed"].astype(int)) != {seed} or set(
            seed_decisions["seed"].astype(int)
        ) != {seed}:
            raise ValueError(f"Seed column mismatch: {seed_dir}")
        _validate_seed_tables(
            seed_metrics,
            seed_decisions,
            seed=seed,
            view_cost=float(config["view_cost"]),
        )
        metrics_frames.append(seed_metrics)
        decision_frames.append(seed_decisions)
        completion.append(record)
    if not completion:
        raise ValueError(f"No completed seed evaluations under {experiment_root}")
    main_budgets = {int(record["main_budget"]) for record in completion}
    view_costs = {float(record["view_cost"]) for record in completion}
    cost_matrices = {
        json.dumps(record["cost_matrix"], sort_keys=True) for record in completion
    }
    statuses = {str(record["evaluation_status"]) for record in completion}
    if len(main_budgets) != 1 or len(view_costs) != 1 or len(cost_matrices) != 1:
        raise ValueError("Seed completions disagree on registered budget or cost config")
    if statuses != {"development_test_consumed"}:
        raise ValueError(f"Unexpected evaluation status: {sorted(statuses)}")
    common_configs = []
    common_inputs = []
    for record in completion:
        config = dict(record["config"])
        config.pop("seed", None)
        common_configs.append(json.dumps(config, sort_keys=True))
        inputs = dict(record["input_sha256"])
        inputs.pop("cache", None)
        inputs.pop("cache_metadata", None)
        common_inputs.append(json.dumps(inputs, sort_keys=True))
    for label, values in (
        ("non-seed config", common_configs),
        ("shared inputs", common_inputs),
        ("source code", [json.dumps(record["source_sha256"], sort_keys=True) for record in completion]),
        ("runtime", [json.dumps(record["runtime_versions"], sort_keys=True) for record in completion]),
        ("input semantics", [json.dumps(record["input_semantics"], sort_keys=True) for record in completion]),
    ):
        if len(set(values)) != 1:
            raise ValueError(f"Seed completions disagree on {label}")
    main_budget = main_budgets.pop()
    view_cost = view_costs.pop()
    metrics = pd.concat(metrics_frames, ignore_index=True)
    decisions = pd.concat(decision_frames, ignore_index=True)
    metric_seeds = tuple(sorted(metrics["seed"].astype(int).unique()))
    decision_seeds = tuple(sorted(decisions["seed"].astype(int).unique()))
    if metric_seeds != EXPECTED_SEEDS or decision_seeds != EXPECTED_SEEDS:
        raise ValueError(f"Expected registered seeds {EXPECTED_SEEDS}")
    sample_sets = [
        frozenset(group["sample_id"].astype(str))
        for _, group in decisions.groupby("seed")
    ]
    if len(set(sample_sets)) != 1:
        raise ValueError("Development-test sample IDs differ across model seeds")
    if (
        decisions.groupby("sample_id")[["target", "spatial_block_id", "sequence_id"]]
        .nunique(dropna=False)
        .to_numpy()
        > 1
    ).any():
        raise ValueError("Development-test sample metadata differs across model seeds")
    value_columns = [
        "accuracy",
        "macro_f1",
        "severe_recall",
        "severe_miss_rate",
        "extreme_error_rate",
        "operational_cost",
        "normalized_operational_cost",
        "nll",
        "additional_acquisition_cost",
        "total_cost",
        "mean_cumulative_action_regret",
    ]
    aggregate = (
        metrics.groupby(["policy", "view_count"])[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part)).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in aggregate.columns
    ]
    aggregate.to_csv(experiment_root / "aggregate_metrics.csv", index=False)
    comparisons = [
        _dependency_cluster_bootstrap_comparison(
            decisions,
            baseline=baseline,
            budget=main_budget,
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        for baseline in (
            "random",
            "clockwise",
            "farthest",
            "max_building_privileged",
            "max_confidence_privileged",
        )
    ]
    pd.DataFrame(comparisons).to_csv(
        experiment_root / "main_budget_comparisons.csv", index=False
    )
    initial = metrics[metrics["view_count"] == 1].groupby("seed")["operational_cost"].mean()
    main = metrics[metrics["view_count"] == main_budget]
    closure_rows = []
    for seed in sorted(main["seed"].unique()):
        oracle_cost = float(
            main[(main["seed"] == seed) & (main["policy"] == "oracle_cost")][
                "operational_cost"
            ].iloc[0]
        )
        denominator = float(initial.loc[seed] - oracle_cost)
        for row in main[main["seed"] == seed].itertuples(index=False):
            closure_rows.append(
                {
                    "seed": seed,
                    "policy": row.policy,
                    "oracle_gap_closure": (
                        float((initial.loc[seed] - row.operational_cost) / denominator)
                        if denominator > 1e-12
                        else np.nan
                    ),
                }
            )
    closure = pd.DataFrame(closure_rows).groupby("policy")["oracle_gap_closure"].agg(["mean", "std"])
    closure_lookup = closure["mean"].to_dict()
    main_lookup = aggregate[aggregate["view_count"] == main_budget].set_index("policy")
    learned = main_lookup.loc["learned"]
    building = main_lookup.loc["max_building_privileged"]
    random_row = main_lookup.loc["random"]
    farthest = main_lookup.loc["farthest"]
    oracle = main_lookup.loc["oracle_cost"]
    building_comparison = next(
        row for row in comparisons if row["baseline"] == "max_building_privileged"
    )
    go = (
        building_comparison["dependency_component_bootstrap_cost_ci95"][0] > 0
        and building_comparison["cost_improvement_positive_seeds"] == 5
        and learned["severe_miss_rate_mean"] < building["severe_miss_rate_mean"]
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "seeds": sorted(int(value) for value in metrics["seed"].unique()),
        "main_budget": main_budget,
        "view_cost": view_cost,
        "test_rows": int(decisions["sample_id"].nunique()),
        "test_spatial_blocks": int(decisions["spatial_block_id"].nunique()),
        "sequence_connected_dependency_components": len(
            set(_sequence_connected_components(decisions).values())
        ),
        "bootstrap_config": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "unit": "sequence-spatial connected dependency component",
        },
        "headline_estimand": "sample-weighted mean, then mean and std across model seeds",
        "comparison_estimand": "equal dependency-component macro mean across model seeds",
        "go_criterion": (
            "building-minus-learned dependency-component bootstrap CI lower bound > 0; "
            "positive sample-weighted improvement in all five seeds; learned severe-miss "
            "rate below building"
        ),
        "summarizer_sha256": _sha256(Path(__file__).resolve()),
        "runner_sha256": runner_sha256,
        "go_criterion_met": bool(go),
        "go_status": "GO" if go else "NO-GO",
        "comparisons": comparisons,
        "oracle_gap_closure": closure.reset_index().to_dict(orient="records"),
        "development_test_artifacts": completion,
    }
    (experiment_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# CVIAN spatial-v1 active next-view benchmark",
        "",
        "**Status:** five-seed frozen-encoder sequential-reveal development experiment "
        "complete; " + ("GO criterion met." if go else "development GO criterion not met."),
        "",
        "## Development protocol",
        "",
        "- 4,121 CVIAN panoramas were georeferenced before splitting; retained roles are "
        "3,253 train, 410 validation, and 415 development test, with 43 boundary exclusions.",
        "- Each 1024x512 equirectangular panorama is represented by eight overlapping "
        "90-degree sectors at 45-degree relative-yaw intervals.",
        "- Every episode starts from post-event overhead plus fixed forward `sector_0`; "
        "the main comparison uses total street-view budget `k=3`.",
        "- Frozen five-seed cross-view encoders produce sector/overhead embeddings. They "
        "were trained on the full spatial train role, so the perception stage is not "
        "role-disjoint. Downstream only, 187 blocks fit the mask-aware classifier and 47 "
        "disjoint blocks fit the selector (seed 42 counts; partitions vary by seed).",
        "- The run fingerprint binds the historical v1 embedding and visibility metadata. "
        "Checkpoint hashes are recorded, but the exact SegFormer revision is unresolved; "
        "that signal is restricted to the privileged development baseline.",
        "- Selector training states contain `k=1..3`, with one-step targets reaching "
        "`k=4`. Learned decisions for endpoints `k=5..8` are budget extrapolations; "
        "the main `k=3` result is within the trained state range.",
        "- Validation selects model epochs and temperature. The 415-row, 6-block spatial "
        "test was inspected during implementation and is now consumed; this result is a "
        "development evaluation, not a confirmatory locked-test claim.",
        "- Online policies see overhead, revealed sector embeddings/logits, the revealed "
        "mask, current uncertainty, step, and budget. Hidden visibility/confidence and "
        "labels never enter the learned observation. State aggregation is reveal-order "
        "invariant under fixed canonical sector indexing, not sector-index permutation "
        "invariant or cyclically equivariant.",
        "- Operational cost matrix is `[[0,1,4],[1,0,1],[8,8,0]]`; each additional "
        "sector costs 0.5. The initial sector is sunk cost.",
        "- Claim scope is offline sequential evidence reveal and `risk-aware`, not real-world "
        "acquisition and not finite-sample `risk-controlled` deployment.",
        "",
        f"## Five-seed development-test result at k={main_budget}",
        "",
        *_markdown_table(aggregate, main_budget),
        "",
        "At equal view budget, the learned selector does not beat the predefined "
        "building-centered privileged heuristic, random selection, or the online "
        "farthest-sector coverage rule. The development GO criterion is therefore **NO-GO**.",
        "",
        f"The learned policy operational cost is {learned['operational_cost_mean']:.3f}, "
        f"versus random {random_row['operational_cost_mean']:.3f}, farthest "
        f"{farthest['operational_cost_mean']:.3f}, building {building['operational_cost_mean']:.3f}, "
        f"and the label-aware oracle {oracle['operational_cost_mean']:.3f}. The oracle result "
        "shows substantial actionable view-selection headroom even though the present learned "
        "selector does not capture it.",
        "",
        f"Relative to the label-aware greedy one-step oracle at `k={main_budget}`, learned closes "
        f"{closure_lookup['learned']:.1%} of the operational-cost gap, random closes "
        f"{closure_lookup['random']:.1%}, farthest closes {closure_lookup['farthest']:.1%}, "
        f"and hidden maximum-confidence closes "
        f"{closure_lookup['max_confidence_privileged']:.1%}.",
        "",
        "The configured acquisition cost is also consequential: two additional views cost "
        "1.0 at `k=3`, larger than the average error-cost reduction even for the oracle. "
        "Thus a fixed policy that acquires two views for every sample is not cost-optimal at "
        "the current ontology cost; a future adaptive stop/defer policy must target only "
        "high-value cases.",
        "",
        "## Statistical reading",
        "",
    ]
    for comparison in comparisons:
        low, high = comparison["dependency_component_bootstrap_cost_ci95"]
        severe_low, severe_high = comparison[
            "dependency_component_bootstrap_severe_miss_ci95"
        ]
        lines.append(
            f"- `{comparison['baseline']}` minus learned cost improvement: "
            f"{comparison['dependency_component_macro_cost_improvement']:+.3f}, "
            f"4-component bootstrap 95% CI "
            f"[{low:+.3f}, {high:+.3f}]; severe-miss improvement "
            f"{comparison['dependency_component_macro_severe_miss_improvement']:+.3f} "
            f"[{severe_low:+.3f}, {severe_high:+.3f}]."
        )
    lines.extend(
        [
            "",
            "The test has six spatial blocks, but cross-block sequence overlap connects "
            "them into only four dependency components. Intervals therefore resample those "
            "four components and remain descriptive; they cannot support a strong "
            "risk-control guarantee. Headline table values are sample-weighted means across "
            "model seeds, whereas comparison centers are equal-component macro means. The "
            "test was consumed during implementation, so future selector variants require "
            "a new sequence/event holdout. Sectors and model seeds are not treated as "
            "independent samples. Random uses one deterministic trajectory per model seed, "
            "so its seed spread also mixes model and policy randomness.",
            "",
            "## Interpretation and next experiment",
            "",
            "1. The active-view task is viable as a benchmark because the greedy one-step "
            "label-aware oracle gap is large; it is not a globally optimal k-step bound.",
            "2. The current supervised action-classification target is weak (validation action "
            "macro-F1 about 0.17-0.20) and does not generalize reliably across seeds.",
            "3. Farthest angular coverage is the strongest admissible simple rule on average; "
            "hidden confidence is a strong privileged signal, suggesting utility regression "
            "with candidate-side features available only after a cheap preview is worth testing.",
            "4. The next model should predict utility with spatial-block OOF targets and add "
            "adaptive `stop/defer_human`; it must be evaluated before any Milton transfer.",
            "5. Spatial-v1 still has sequence overlap, so a sequence-grouped sensitivity run "
            "is required before claiming capture-session generalization.",
            "",
            "Machine-readable outputs are under "
            "`outputs/analysis/active_view_cvian_spatial_v1/experiment/`.",
        ]
    )
    output_doc.parent.mkdir(parents=True, exist_ok=True)
    output_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"go": go, "output": str(output_doc)}, indent=2))


if __name__ == "__main__":
    main()
