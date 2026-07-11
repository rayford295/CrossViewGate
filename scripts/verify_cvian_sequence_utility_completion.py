"""Independently verify the completed CVIAN sequence-utility evaluation.

This is a post-consumption, hash-only verifier.  It deliberately does not
import the evaluation runner, deserialize an NPZ cache, load a checkpoint, or
invoke model inference.  The only scored data it reads semantically are the
already-saved per-sample decision CSVs, from which it independently recomputes
the frozen primary paired-component analysis.

The verifier also handles the one known Windows serialization defect.  Python
translated the evaluator's canonical LF commitment rendering to CRLF while
writing it.  That difference is tolerated *only* when the parsed JSON has a
valid internal canonical fingerprint, CRLF-to-LF normalization exactly
recovers the declared rendering, and every downstream attestation references
the actual on-disk commitment SHA-256.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SCHEMA = "cvian-sequence-utility-protocol-v1"
FIT_SCHEMA = "cvian-sequence-utility-fit-v1"
COMMITMENT_SCHEMA = "cvian-sequence-utility-evaluation-commitment-v1"
EVALUATION_SCHEMA = "cvian-sequence-utility-evaluation-v1"
AGGREGATE_SCHEMA = "cvian-sequence-utility-primary-aggregate-v1"
AGGREGATE_COMPLETION_SCHEMA = (
    "cvian-sequence-utility-primary-aggregate-completion-v1"
)
REGISTRY_COMPLETION_SCHEMA = "cvian-sequence-utility-registry-completion-v1"
VERIFICATION_SCHEMA = "cvian-sequence-utility-completion-verification-v1"

EXPECTED_SEEDS = (42, 123, 456, 789, 1011)
ROLE_NAMES = ("base_fit", "selector_fit", "validation", "prospective_test")
FIT_ROLES = ROLE_NAMES[:3]
PRIMARY_UTILITY_POLICY = "utility_regression"
PRIMARY_BASELINES = ("farthest", "max_building_privileged")

SOURCE_FILES = {
    "runner": "scripts/run_cvian_sequence_utility_experiment.py",
    "active_view": "crossview_conflict/decision/active_view.py",
    "active_view_utility": "crossview_conflict/decision/active_view_utility.py",
    "artifact_provenance": "crossview_conflict/decision/artifact_provenance.py",
}
FIT_ARTIFACT_FIELDS = {
    "fit_artifact_sha256": "fit_artifact.pt",
    "training_history_sha256": "training_history.json",
    "validation_regret_sha256": "validation_top1_regret.csv",
    "stop_policy_status_sha256": "stop_policy_status.csv",
}
FIT_FINGERPRINT_FIELDS = (
    "schema_version",
    "seed",
    "config_sha256",
    "protocol_summary_sha256",
    "test_commitment_sha256",
    "fit_manifest_sha256",
    "input_provenance",
    "source_sha256",
    "git_head",
    "runtime_versions",
    "base_runtime_lock_provenance",
    "hyperparameters",
    "prospective_test_loaded",
)


class VerificationError(ValueError):
    """Raised when a declared completion edge does not verify."""


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise VerificationError(f"Required artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected a JSON object: {path}")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise VerificationError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _safe_report_path(repo_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise VerificationError("Commitment does not declare a primary report path")
    candidate = (repo_root / Path(relative)).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise VerificationError("Primary report path escapes the repository") from exc
    return candidate


def _validate_config(config: Mapping[str, Any]) -> None:
    _expect(config.get("schema_version"), CONFIG_SCHEMA, "config schema")
    _expect(tuple(config.get("seeds", ())), EXPECTED_SEEDS, "registered seeds")
    _expect(config.get("main_fixed_budget"), 3, "primary fixed budget")
    _expect(
        config.get("random_trajectories_per_model_seed"),
        32,
        "random trajectory count",
    )
    policies = set(config.get("fixed_budget_policies", ()))
    required = {PRIMARY_UTILITY_POLICY, *PRIMARY_BASELINES}
    if not required.issubset(policies):
        raise VerificationError("Config is missing a frozen primary policy")
    criterion = config.get("primary_go_criterion")
    if not isinstance(criterion, dict):
        raise VerificationError("Config has no primary GO criterion")
    bootstrap = criterion.get("paired_bootstrap")
    expected_bootstrap = {
        "resampling_unit": "dependency_component",
        "pairing": (
            "resample each shared component with all five model-seed and "
            "paired-policy values intact"
        ),
        "resamples": 10000,
        "random_seed": 42,
        "confidence_level": 0.95,
        "interval": "percentile",
    }
    _expect(bootstrap, expected_bootstrap, "paired bootstrap registration")
    _expect(
        criterion.get("severe_miss_noninferiority_margin"),
        0.0,
        "severe-miss margin",
    )
    _expect(
        config.get("external_confirmation_required_for_strong_claim"),
        True,
        "external-confirmation requirement",
    )


def _validate_commitment_serialization(
    path: Path, commitment: Mapping[str, Any]
) -> tuple[str, str | None, str]:
    """Return raw SHA, optional defect, and canonical-LF rendered SHA."""
    raw = path.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    canonical_lf = (json.dumps(dict(commitment), indent=2) + "\n").encode("utf-8")
    canonical_lf_sha = hashlib.sha256(canonical_lf).hexdigest()
    if raw == canonical_lf:
        return raw_sha, None, canonical_lf_sha
    # This is the sole serialization exception.  It is deliberately exact:
    # arbitrary whitespace or key-order rewrites are not accepted.
    if b"\r\n" in raw and raw.replace(b"\r\n", b"\n") == canonical_lf:
        return raw_sha, "windows_newline_hash_mismatch", canonical_lf_sha
    raise VerificationError(
        "Evaluation commitment bytes differ from the canonical rendering for a "
        "reason other than CRLF translation"
    )


def _protocol_registration(
    protocol_dir: Path,
    summary: Mapping[str, Any],
    test_commitment: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    _expect(
        summary.get("schema_version"),
        "cvian-sequence-four-role-summary-v1",
        "protocol summary schema",
    )
    _expect(
        test_commitment.get("schema_version"),
        "cvian-sequence-test-commitment-v1",
        "test commitment schema",
    )
    role_rows = summary.get("role_rows")
    if not isinstance(role_rows, dict) or set(role_rows) != set(ROLE_NAMES):
        raise VerificationError("Protocol role-row registration is incomplete")
    registered = summary.get("role_manifest_sha256")
    if not isinstance(registered, dict) or set(registered) != {
        f"{role}.csv" for role in ROLE_NAMES
    }:
        raise VerificationError("Protocol role-manifest registration is incomplete")
    role_hashes: dict[str, str] = {}
    for role in ROLE_NAMES:
        actual = _sha256(protocol_dir / f"{role}.csv")
        _expect(registered[f"{role}.csv"], actual, f"{role} manifest SHA-256")
        role_hashes[role] = actual
    _expect(test_commitment.get("role"), "prospective_test", "test role")
    _expect(
        test_commitment.get("manifest_sha256"),
        role_hashes["prospective_test"],
        "test commitment manifest SHA-256",
    )
    _expect(
        test_commitment.get("row_count"),
        role_rows["prospective_test"],
        "test commitment row count",
    )
    status = "selector_selection_holdout_with_historical_base_exposure"
    _expect(test_commitment.get("status"), status, "test commitment status")
    summary_status = summary.get("test_status")
    if not isinstance(summary_status, dict):
        raise VerificationError("Protocol summary has no test-status object")
    _expect(summary_status.get("status"), status, "protocol test status")
    _expect(
        summary_status.get("new_protocol_test_scored"),
        False,
        "protocol pre-evaluation scoring flag",
    )
    compact = {
        "protocol_summary_sha256": _sha256(protocol_dir / "protocol_summary.json"),
        "test_commitment_sha256": _sha256(protocol_dir / "test_commitment.json"),
        "role_manifest_sha256": role_hashes,
        "role_rows": role_rows,
        "validated_manifest_roles": list(ROLE_NAMES),
        "validated_manifest_sha256": role_hashes,
    }
    return role_hashes, _canonical_hash(compact)


def _sample_average_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "policy",
        "scope",
        "sample_id",
        "sequence_id",
        "spatial_block_id",
        "dependency_component",
        "trajectory",
        "target",
        "operational_cost",
        "severe_miss",
        "view_count",
        "terminal_action",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise VerificationError(f"Decision frame lacks columns: {sorted(missing)}")
    numeric = frame[["operational_cost", "severe_miss", "view_count"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise VerificationError("Decision metric columns must be finite")
    if (frame["operational_cost"].to_numpy(dtype=float) < 0.0).any():
        raise VerificationError("Operational cost cannot be negative")
    severe = frame["severe_miss"].to_numpy(dtype=float)
    if ((severe < 0.0) | (severe > 1.0)).any():
        raise VerificationError("Severe-miss indicators must lie in [0, 1]")
    duplicate_key = ["seed", "policy", "scope", "sample_id", "trajectory"]
    if frame.duplicated(duplicate_key).any():
        raise VerificationError("Duplicate sample/trajectory decision row")
    key = [
        "seed",
        "policy",
        "scope",
        "sample_id",
        "sequence_id",
        "spatial_block_id",
        "dependency_component",
        "target",
    ]
    work = frame.copy()
    work["defer_probability"] = (work["terminal_action"] == "defer").astype(float)
    collapsed = (
        work.groupby(key, sort=True, as_index=False)
        .agg(
            operational_cost=("operational_cost", "mean"),
            severe_miss=("severe_miss", "mean"),
            view_count=("view_count", "mean"),
            defer_probability=("defer_probability", "mean"),
            trajectory_count=("trajectory", "nunique"),
        )
        .reset_index(drop=True)
    )
    if len(collapsed) != frame["sample_id"].nunique():
        raise VerificationError("Sample identifiers are not unique within a policy/seed")
    return collapsed


def _paired_component_bootstrap(
    contrasts: pd.DataFrame,
    *,
    value_column: str,
    resamples: int,
    random_seed: int,
    confidence_level: float,
) -> tuple[float, float]:
    pivot = contrasts.pivot(
        index="dependency_component", columns="seed", values=value_column
    ).sort_index(axis=0).sort_index(axis=1)
    if tuple(int(value) for value in pivot.columns) != EXPECTED_SEEDS:
        raise VerificationError("Bootstrap does not contain the five locked seeds")
    if pivot.empty or pivot.isna().any().any():
        raise VerificationError("Bootstrap contrasts are incomplete")
    values = pivot.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    sampled = rng.integers(0, len(values), size=(resamples, len(values)), endpoint=False)
    replicates = values[sampled].mean(axis=(1, 2))
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(replicates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def recompute_primary_analysis(
    decisions: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Independent implementation of the frozen five-seed GO analysis."""
    _validate_config(config)
    primary = decisions[decisions["scope"] == "fixed_k3_primary"].copy()
    if set(int(value) for value in primary["seed"].unique()) != set(EXPECTED_SEEDS):
        raise VerificationError("Primary decisions lack exactly the five locked seeds")
    required = {PRIMARY_UTILITY_POLICY, *PRIMARY_BASELINES}
    if not required.issubset(set(primary["policy"].unique())):
        raise VerificationError("Primary decisions are missing a GO policy")

    bootstrap = config["primary_go_criterion"]["paired_bootstrap"]
    csv_rows: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    common_component_set: set[str] | None = None
    sample_counts: dict[str, int] = {}

    for baseline in PRIMARY_BASELINES:
        component_rows: list[pd.DataFrame] = []
        per_seed: dict[str, Any] = {}
        for seed in EXPECTED_SEEDS:
            utility = _sample_average_trajectories(
                primary[
                    (primary["seed"] == seed)
                    & (primary["policy"] == PRIMARY_UTILITY_POLICY)
                ]
            )
            baseline_frame = _sample_average_trajectories(
                primary[
                    (primary["seed"] == seed) & (primary["policy"] == baseline)
                ]
            )
            if not (utility["trajectory_count"] == 1).all() or not (
                baseline_frame["trajectory_count"] == 1
            ).all():
                raise VerificationError("Primary policies must have one trajectory")
            join_keys = [
                "seed",
                "scope",
                "sample_id",
                "sequence_id",
                "spatial_block_id",
                "dependency_component",
                "target",
            ]
            utility = utility.drop(columns=["policy"]).rename(
                columns={
                    "operational_cost": "utility_operational_cost",
                    "severe_miss": "utility_severe_miss",
                }
            )
            baseline_frame = baseline_frame.drop(columns=["policy"]).rename(
                columns={
                    "operational_cost": "baseline_operational_cost",
                    "severe_miss": "baseline_severe_miss",
                }
            )
            paired = utility.merge(
                baseline_frame,
                on=join_keys,
                how="inner",
                validate="one_to_one",
                suffixes=("_utility", "_baseline"),
            )
            if len(paired) == 0 or len(paired) != len(utility) or len(paired) != len(
                baseline_frame
            ):
                raise VerificationError(f"Unpaired {baseline} decisions for seed {seed}")
            sample_counts[str(seed)] = int(len(paired))
            paired["cost_difference_baseline_minus_utility"] = (
                paired["baseline_operational_cost"]
                - paired["utility_operational_cost"]
            )
            paired["severe_difference_baseline_minus_utility"] = (
                paired["baseline_severe_miss"] - paired["utility_severe_miss"]
            )
            component = (
                paired.groupby("dependency_component", sort=True, as_index=False)
                .agg(
                    samples=("sample_id", "size"),
                    utility_component_mean_cost=("utility_operational_cost", "mean"),
                    baseline_component_mean_cost=(
                        "baseline_operational_cost",
                        "mean",
                    ),
                    cost_difference_baseline_minus_utility=(
                        "cost_difference_baseline_minus_utility",
                        "mean",
                    ),
                    utility_component_mean_severe_miss=(
                        "utility_severe_miss",
                        "mean",
                    ),
                    baseline_component_mean_severe_miss=(
                        "baseline_severe_miss",
                        "mean",
                    ),
                    severe_difference_baseline_minus_utility=(
                        "severe_difference_baseline_minus_utility",
                        "mean",
                    ),
                )
                .assign(seed=seed)
            )
            component_set = set(component["dependency_component"].astype(str))
            if common_component_set is None:
                common_component_set = component_set
            elif component_set != common_component_set:
                raise VerificationError("Dependency components changed across seeds")
            component_rows.append(component)
            cost_difference = float(
                component["cost_difference_baseline_minus_utility"].mean()
            )
            severe_difference = float(
                component["severe_difference_baseline_minus_utility"].mean()
            )
            seed_result = {
                "samples": int(component["samples"].sum()),
                "dependency_components": int(len(component)),
                "utility_component_macro_operational_cost": float(
                    component["utility_component_mean_cost"].mean()
                ),
                "baseline_component_macro_operational_cost": float(
                    component["baseline_component_mean_cost"].mean()
                ),
                "cost_difference_baseline_minus_utility": cost_difference,
                "utility_component_macro_severe_miss_rate": float(
                    component["utility_component_mean_severe_miss"].mean()
                ),
                "baseline_component_macro_severe_miss_rate": float(
                    component["baseline_component_mean_severe_miss"].mean()
                ),
                "severe_difference_baseline_minus_utility": severe_difference,
                "utility_lower_cost": bool(cost_difference > 0.0),
            }
            per_seed[str(seed)] = seed_result
            csv_rows.append(
                {
                    "row_type": "per_seed",
                    "baseline": baseline,
                    "seed": str(seed),
                    **seed_result,
                    "bootstrap_ci_lower": math.nan,
                    "bootstrap_ci_upper": math.nan,
                    "all_five_seeds_lower_cost": math.nan,
                    "severe_miss_no_worse": math.nan,
                    "comparison_go": math.nan,
                }
            )

        all_components = pd.concat(component_rows, ignore_index=True)
        ci_lower, ci_upper = _paired_component_bootstrap(
            all_components,
            value_column="cost_difference_baseline_minus_utility",
            resamples=int(bootstrap["resamples"]),
            random_seed=int(bootstrap["random_seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )
        overall_cost_difference = float(
            all_components["cost_difference_baseline_minus_utility"].mean()
        )
        overall_severe_difference = float(
            all_components["severe_difference_baseline_minus_utility"].mean()
        )
        all_five_lower = bool(
            all(item["utility_lower_cost"] for item in per_seed.values())
        )
        margin = float(
            config["primary_go_criterion"]["severe_miss_noninferiority_margin"]
        )
        severe_no_worse = bool(overall_severe_difference >= -margin)
        ci_positive = bool(ci_lower > 0.0)
        comparison_go = bool(all_five_lower and ci_positive and severe_no_worse)
        overall = {
            "dependency_components": int(
                all_components["dependency_component"].nunique()
            ),
            "utility_component_macro_operational_cost": float(
                all_components["utility_component_mean_cost"].mean()
            ),
            "baseline_component_macro_operational_cost": float(
                all_components["baseline_component_mean_cost"].mean()
            ),
            "cost_difference_baseline_minus_utility": overall_cost_difference,
            "bootstrap_95_percentile_ci_lower": ci_lower,
            "bootstrap_95_percentile_ci_upper": ci_upper,
            "utility_component_macro_severe_miss_rate": float(
                all_components["utility_component_mean_severe_miss"].mean()
            ),
            "baseline_component_macro_severe_miss_rate": float(
                all_components["baseline_component_mean_severe_miss"].mean()
            ),
            "severe_difference_baseline_minus_utility": overall_severe_difference,
            "all_five_seeds_lower_cost": all_five_lower,
            "bootstrap_ci_lower_strictly_positive": ci_positive,
            "severe_miss_no_worse": severe_no_worse,
            "comparison_go": comparison_go,
        }
        comparisons[baseline] = {"per_seed": per_seed, "overall": overall}
        csv_rows.append(
            {
                "row_type": "all_seed_component_macro",
                "baseline": baseline,
                "seed": "ALL",
                "samples": int(sum(sample_counts.values())),
                "dependency_components": overall["dependency_components"],
                "utility_component_macro_operational_cost": overall[
                    "utility_component_macro_operational_cost"
                ],
                "baseline_component_macro_operational_cost": overall[
                    "baseline_component_macro_operational_cost"
                ],
                "cost_difference_baseline_minus_utility": overall_cost_difference,
                "utility_component_macro_severe_miss_rate": overall[
                    "utility_component_macro_severe_miss_rate"
                ],
                "baseline_component_macro_severe_miss_rate": overall[
                    "baseline_component_macro_severe_miss_rate"
                ],
                "severe_difference_baseline_minus_utility": overall_severe_difference,
                "utility_lower_cost": bool(overall_cost_difference > 0.0),
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "all_five_seeds_lower_cost": all_five_lower,
                "severe_miss_no_worse": severe_no_worse,
                "comparison_go": comparison_go,
            }
        )

    aggregate_go = bool(
        all(comparisons[name]["overall"]["comparison_go"] for name in PRIMARY_BASELINES)
    )
    criterion = config["primary_go_criterion"]
    analysis = {
        "schema_version": AGGREGATE_SCHEMA,
        "claim_scope": config["claim_scope"],
        "test_status": config["test_status"],
        "primary_scope": "fixed_k3_primary",
        "utility_policy": PRIMARY_UTILITY_POLICY,
        "baselines": list(PRIMARY_BASELINES),
        "model_seeds": list(EXPECTED_SEEDS),
        "sample_count_per_seed": sample_counts,
        "dependency_components": len(common_component_set or set()),
        "estimand": criterion["estimand"],
        "contrast": criterion["contrast"],
        "severe_miss_estimand": criterion["severe_miss_estimand"],
        "severe_miss_definition": criterion["severe_miss_definition"],
        "severe_miss_noninferiority_margin": criterion[
            "severe_miss_noninferiority_margin"
        ],
        "severe_miss_rule": criterion["severe_miss_rule"],
        "paired_bootstrap": copy.deepcopy(bootstrap),
        "comparisons": comparisons,
        "requirements": copy.deepcopy(criterion["requirements"]),
        "go": aggregate_go,
        "decision": "GO" if aggregate_go else "NO-GO",
        "adaptive_results_in_primary_decision": False,
        "external_confirmation_required_for_strong_claim": bool(
            config["external_confirmation_required_for_strong_claim"]
        ),
    }
    return analysis, pd.DataFrame(csv_rows)


def render_primary_report(
    analysis: Mapping[str, Any], comparisons: pd.DataFrame
) -> str:
    """Independently reproduce the frozen machine-derived Markdown report."""
    lines = [
        "# CVIAN sequence utility experiment v1",
        "",
        f"**Primary decision: {analysis['decision']}.**",
        "",
        f"Claim scope: {analysis['claim_scope']}.",
        "",
        (
            "This report is generated only after all five one-time prospective-test "
            "seed artifacts have been staged. The primary estimand is equal "
            "dependency-component macro operational cost at fixed k=3; adaptive "
            "results are secondary and cannot change the decision."
        ),
        "",
        "## Frozen primary comparisons",
        "",
        (
            "Positive cost and severe-miss differences favor utility_regression "
            "(baseline minus utility)."
        ),
        "",
        "| Baseline | Mean cost difference | Paired component bootstrap 95% CI | Five seeds lower | Severe miss no worse | Result |",
        "|---|---:|---:|:---:|:---:|:---:|",
    ]
    for baseline in PRIMARY_BASELINES:
        overall = analysis["comparisons"][baseline]["overall"]
        lines.append(
            "| {baseline} | {difference:.6f} | [{lower:.6f}, {upper:.6f}] | "
            "{five} | {severe} | {result} |".format(
                baseline=baseline,
                difference=overall["cost_difference_baseline_minus_utility"],
                lower=overall["bootstrap_95_percentile_ci_lower"],
                upper=overall["bootstrap_95_percentile_ci_upper"],
                five="yes" if overall["all_five_seeds_lower_cost"] else "no",
                severe="yes" if overall["severe_miss_no_worse"] else "no",
                result="GO" if overall["comparison_go"] else "NO-GO",
            )
        )
    lines.extend(
        [
            "",
            "## Per-seed component-macro contrasts",
            "",
            "| Baseline | Seed | Utility cost | Baseline cost | Baseline - utility | Utility lower |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    per_seed = comparisons[comparisons["row_type"] == "per_seed"]
    for row in per_seed.to_dict(orient="records"):
        lines.append(
            "| {baseline} | {seed} | {utility:.6f} | {baseline_cost:.6f} | "
            "{difference:.6f} | {lower} |".format(
                baseline=row["baseline"],
                seed=row["seed"],
                utility=row["utility_component_macro_operational_cost"],
                baseline_cost=row["baseline_component_macro_operational_cost"],
                difference=row["cost_difference_baseline_minus_utility"],
                lower="yes" if row["utility_lower_cost"] else "no",
            )
        )
    bootstrap = analysis["paired_bootstrap"]
    lines.extend(
        [
            "",
            "## Decision rule and interpretation",
            "",
            (
                f"The locked percentile bootstrap uses {bootstrap['resamples']:,} "
                f"dependency-component resamples with random seed "
                f"{bootstrap['random_seed']}. Each sampled component retains all "
                "five model-seed and paired-policy values."
            ),
            "",
            (
                "GO requires both baselines to pass all three requirements: a "
                "strictly positive component-macro contrast in every seed, a "
                "strictly positive 95% CI lower bound, and a nonnegative "
                "baseline-minus-utility severe-miss contrast."
            ),
            "",
            (
                "For random_mc32 descriptive metrics, trajectories are averaged "
                "within sample before dependency-component aggregation."
            ),
            "",
            (
                "Adaptive STOP/ACQUIRE/DEFER results are secondary heuristics. "
                "No formal STOP calibration was performed because the current "
                "protocol has no independent policy-trajectory calibration role; "
                "the fail-closed variant disables STOP and the threshold-1 variant "
                "is risk-aware, not risk-controlled."
            ),
            "",
            (
                "This is a within-CVIAN development confirmation with historical "
                "base exposure, not external confirmation. It evaluates the "
                "label-cost Phase-1 baseline and does not complete the separate "
                "attestation-coverage claim."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _semantic_json_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if _canonical_hash(actual) != _canonical_hash(expected):
        raise VerificationError("Stored aggregate JSON differs from recomputation")


def _normalized_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8").replace("\r\n", "\n")
    except (OSError, UnicodeError) as exc:
        raise VerificationError(f"Invalid UTF-8 artifact: {path}") from exc


def verify_completion(
    *,
    repo_root: Path = REPO_ROOT,
    config_path: Path | None = None,
    protocol_dir: Path | None = None,
    cache_root: Path | None = None,
    output_root: Path | None = None,
    visibility_dir: Path | None = None,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Verify the complete declared artifact DAG without rescoring anything."""
    repo_root = repo_root.resolve()
    config_path = (config_path or repo_root / "configs/cvian_sequence_utility_v1.json").resolve()
    protocol_dir = (
        protocol_dir
        or repo_root / "data/splits/ian_hurricane_sequence_four_role_v1"
    ).resolve()
    cache_root = (
        cache_root or repo_root / "outputs/cvian_sequence_active_v2/cache"
    ).resolve()
    visibility_dir = (
        visibility_dir or repo_root / "outputs/cvian_sequence_active_v2/visibility"
    ).resolve()

    config = _load_json(config_path)
    _validate_config(config)
    execution = config.get("execution_lock")
    if not isinstance(execution, dict):
        raise VerificationError("Config has no execution lock")
    locked_output = _resolve(repo_root, execution["evaluation_output_root"])
    output_root = (output_root or locked_output).resolve()
    _expect(output_root, locked_output, "locked evaluation output root")
    locked_registry = _resolve(repo_root, execution["evaluation_registry_root"])
    registry_root = (registry_root or locked_registry).resolve()
    _expect(registry_root, locked_registry, "locked evaluation registry root")

    summary_path = protocol_dir / "protocol_summary.json"
    test_commitment_path = protocol_dir / "test_commitment.json"
    summary = _load_json(summary_path)
    test_commitment = _load_json(test_commitment_path)
    role_hashes, protocol_registration_sha = _protocol_registration(
        protocol_dir, summary, test_commitment
    )
    protocol_summary_sha = _sha256(summary_path)
    test_commitment_sha = _sha256(test_commitment_path)

    registry_dir = registry_root / test_commitment_sha
    commitment_path = registry_dir / "evaluation_commitment.json"
    started_path = registry_dir / "evaluation_started.json"
    registry_completion_path = registry_dir / "evaluation_complete.json"
    commitment = _load_json(commitment_path)
    _expect(commitment.get("schema_version"), COMMITMENT_SCHEMA, "commitment schema")
    fingerprint = commitment.get("evaluation_fingerprint_sha256")
    if not isinstance(fingerprint, str):
        raise VerificationError("Commitment has no evaluation fingerprint")
    fingerprint_payload = {
        key: value
        for key, value in commitment.items()
        if key != "evaluation_fingerprint_sha256"
    }
    _expect(
        fingerprint,
        _canonical_hash(fingerprint_payload),
        "internal canonical commitment fingerprint",
    )
    commitment_sha, newline_defect, canonical_lf_sha = (
        _validate_commitment_serialization(commitment_path, commitment)
    )

    # Declared input edges.
    _expect(commitment.get("config_sha256"), _sha256(config_path), "config SHA-256")
    _expect(
        commitment.get("protocol_summary_sha256"),
        protocol_summary_sha,
        "protocol summary SHA-256",
    )
    _expect(
        commitment.get("test_commitment_sha256"),
        test_commitment_sha,
        "protocol test commitment SHA-256",
    )
    _expect(
        commitment.get("prospective_test_manifest_sha256"),
        role_hashes["prospective_test"],
        "prospective-test manifest SHA-256",
    )
    _expect(
        Path(str(commitment.get("locked_output_root"))).resolve(),
        output_root,
        "commitment output root",
    )
    _expect(
        Path(str(commitment.get("global_registry_root"))).resolve(),
        registry_root,
        "commitment registry root",
    )
    source_hashes = commitment.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_FILES):
        raise VerificationError("Commitment evaluator source map is incomplete")
    for name, relative in SOURCE_FILES.items():
        _expect(source_hashes[name], _sha256(repo_root / relative), f"source {name}")

    primary_registration = commitment.get("primary_aggregate")
    if not isinstance(primary_registration, dict):
        raise VerificationError("Commitment has no primary aggregate registration")
    _expect(primary_registration.get("schema_version"), AGGREGATE_SCHEMA, "aggregate schema")
    _expect(
        primary_registration.get("utility_policy"),
        PRIMARY_UTILITY_POLICY,
        "primary utility policy",
    )
    _expect(
        tuple(primary_registration.get("baselines", ())),
        PRIMARY_BASELINES,
        "primary baselines",
    )
    _expect(
        primary_registration.get("paired_bootstrap"),
        config["primary_go_criterion"]["paired_bootstrap"],
        "commitment bootstrap",
    )
    report_path = _safe_report_path(repo_root, primary_registration.get("report_path"))

    started = _load_json(started_path)
    started_sha = _sha256(started_path)
    _expect(started.get("schema_version"), COMMITMENT_SCHEMA, "started schema")
    _expect(started.get("evaluation_fingerprint_sha256"), fingerprint, "started fingerprint")
    _expect(
        started.get("evaluation_commitment_sha256"),
        commitment_sha,
        "started actual commitment SHA-256",
    )
    _expect(
        started.get("prospective_test_manifest_sha256"),
        role_hashes["prospective_test"],
        "started test manifest SHA-256",
    )
    _expect(started.get("status"), "prospective_test_scoring_started", "started status")
    _expect(started.get("rerun_if_incomplete_permitted"), False, "rerun permission")

    visibility_metadata_path = visibility_dir / "visibility_metadata.json"
    visibility_metadata_sha = _sha256(visibility_metadata_path)
    visibility_metadata = _load_json(visibility_metadata_path)
    visibility_hashes: dict[str, str] = {}
    for role in ROLE_NAMES:
        path = visibility_dir / f"{role}.npz"
        actual = _sha256(path)
        visibility_hashes[role] = actual
        entry = visibility_metadata.get("splits", {}).get(role, {})
        _expect(entry.get("sha256"), actual, f"visibility {role} metadata SHA-256")
    visibility_provenance = commitment.get("visibility_provenance")
    if not isinstance(visibility_provenance, dict):
        raise VerificationError("Commitment has no visibility provenance")
    _expect(
        commitment.get("visibility_sha256"),
        visibility_hashes["prospective_test"],
        "prospective-test visibility SHA-256",
    )
    _expect(
        visibility_provenance.get("protocol_registration_sha256"),
        protocol_registration_sha,
        "visibility protocol registration",
    )
    _expect(
        visibility_provenance.get("visibility_metadata_sha256"),
        visibility_metadata_sha,
        "visibility metadata SHA-256",
    )
    _expect(
        visibility_provenance.get("visibility_artifacts_sha256"),
        _canonical_hash(visibility_hashes),
        "visibility artifact-set SHA-256",
    )
    _expect(
        visibility_provenance.get("prospective_test_visibility_sha256"),
        visibility_hashes["prospective_test"],
        "visibility prospective-test SHA-256",
    )
    visibility_compact = {
        "protocol_registration_sha256": protocol_registration_sha,
        "visibility_metadata_sha256": visibility_metadata_sha,
        "visibility_artifact_sha256": visibility_hashes,
        "segformer_revision": visibility_metadata.get("model_revision_resolved"),
    }
    _expect(
        visibility_provenance.get("visibility_provenance_sha256"),
        _canonical_hash(visibility_compact),
        "visibility provenance fingerprint",
    )

    seed_inputs = commitment.get("seed_inputs")
    if not isinstance(seed_inputs, dict) or set(seed_inputs) != {
        str(seed) for seed in EXPECTED_SEEDS
    }:
        raise VerificationError("Commitment seed-input map is incomplete")
    decisions: list[pd.DataFrame] = []
    seed_completion_hashes: dict[str, str] = {}
    seed_output_summary: dict[str, Any] = {}

    for seed in EXPECTED_SEEDS:
        seed_key = str(seed)
        declared = seed_inputs[seed_key]
        if not isinstance(declared, dict):
            raise VerificationError(f"Invalid seed-input declaration: {seed}")
        cache_dir = cache_root / f"seed{seed}"
        cache_metadata_path = cache_dir / "cache_metadata.json"
        cache_metadata_sha = _sha256(cache_metadata_path)
        cache_metadata = _load_json(cache_metadata_path)
        _expect(cache_metadata.get("seed"), seed, f"cache metadata seed {seed}")
        _expect(
            declared.get("cache_metadata_sha256"),
            cache_metadata_sha,
            f"seed {seed} cache metadata SHA-256",
        )
        cache_hashes: dict[str, str] = {}
        for role in ROLE_NAMES:
            actual = _sha256(cache_dir / f"{role}.npz")
            cache_hashes[role] = actual
            entry = cache_metadata.get("splits", {}).get(role, {})
            _expect(entry.get("sha256"), actual, f"seed {seed} {role} cache SHA-256")
            _expect(
                cache_metadata.get("source_manifest_sha256", {}).get(role),
                role_hashes[role],
                f"seed {seed} {role} source manifest",
            )
        _expect(
            declared.get("prospective_test_cache_sha256"),
            cache_hashes["prospective_test"],
            f"seed {seed} prospective-test cache SHA-256",
        )

        seed_dir = output_root / f"seed{seed}"
        fit_path = seed_dir / "fit_complete.json"
        fit_sha = _sha256(fit_path)
        fit = _load_json(fit_path)
        _expect(declared.get("fit_completion_sha256"), fit_sha, f"seed {seed} fit completion")
        _expect(fit.get("schema_version"), FIT_SCHEMA, f"seed {seed} fit schema")
        _expect(fit.get("seed"), seed, f"fit seed {seed}")
        _expect(fit.get("prospective_test_loaded"), False, f"seed {seed} fit test-load flag")
        _expect(fit.get("config_sha256"), commitment["config_sha256"], f"seed {seed} fit config")
        _expect(
            fit.get("protocol_summary_sha256"),
            protocol_summary_sha,
            f"seed {seed} fit protocol summary",
        )
        _expect(
            fit.get("test_commitment_sha256"),
            test_commitment_sha,
            f"seed {seed} fit test commitment",
        )
        _expect(fit.get("source_sha256"), source_hashes, f"seed {seed} fit sources")
        _expect(
            fit.get("fit_manifest_sha256"),
            {role: role_hashes[role] for role in FIT_ROLES},
            f"seed {seed} fit manifests",
        )
        fit_fp_payload = {field: fit.get(field) for field in FIT_FINGERPRINT_FIELDS}
        _expect(
            fit.get("fit_fingerprint_sha256"),
            _canonical_hash(fit_fp_payload),
            f"seed {seed} internal fit fingerprint",
        )
        input_provenance = fit.get("input_provenance")
        if not isinstance(input_provenance, dict):
            raise VerificationError(f"Seed {seed} fit has no input provenance")
        _expect(
            input_provenance.get("cache_metadata_sha256"),
            cache_metadata_sha,
            f"seed {seed} fit cache metadata input",
        )
        _expect(
            input_provenance.get("cache_sha256"),
            {role: cache_hashes[role] for role in FIT_ROLES},
            f"seed {seed} fit-role cache inputs",
        )
        for field, filename in FIT_ARTIFACT_FIELDS.items():
            actual = _sha256(seed_dir / filename)
            _expect(fit.get(field), actual, f"seed {seed} {filename}")
        _expect(
            declared.get("fit_artifact_sha256"),
            _sha256(seed_dir / "fit_artifact.pt"),
            f"seed {seed} committed fit artifact",
        )

        completion_path = seed_dir / "evaluation_complete.json"
        completion_sha = _sha256(completion_path)
        completion = _load_json(completion_path)
        _expect(completion.get("schema_version"), EVALUATION_SCHEMA, f"seed {seed} evaluation schema")
        _expect(completion.get("seed"), seed, f"evaluation seed {seed}")
        _expect(completion.get("evaluation_fingerprint_sha256"), fingerprint, f"seed {seed} evaluation fingerprint")
        _expect(completion.get("evaluation_commitment_sha256"), commitment_sha, f"seed {seed} actual commitment SHA-256")
        _expect(completion.get("evaluation_started_sha256"), started_sha, f"seed {seed} started SHA-256")
        _expect(completion.get("fit_completion_sha256"), fit_sha, f"seed {seed} fit edge")
        _expect(completion.get("prospective_test_cache_sha256"), cache_hashes["prospective_test"], f"seed {seed} test-cache edge")
        metrics_path = seed_dir / "metrics.csv"
        decision_path = seed_dir / "per_sample_decisions.csv"
        metrics_sha = _sha256(metrics_path)
        decision_sha = _sha256(decision_path)
        _expect(completion.get("metrics_sha256"), metrics_sha, f"seed {seed} metrics SHA-256")
        _expect(completion.get("decisions_sha256"), decision_sha, f"seed {seed} decisions SHA-256")
        seed_completion_hashes[seed_key] = completion_sha

        # Reading a completed decisions CSV is a post-consumption audit, not a
        # second scoring pass.  No cache or checkpoint is deserialized here.
        frame = pd.read_csv(
            decision_path,
            dtype={
                "policy": str,
                "scope": str,
                "sample_id": str,
                "sequence_id": str,
                "spatial_block_id": str,
                "dependency_component": str,
            },
        )
        if set(frame["seed"].astype(int).unique()) != {seed}:
            raise VerificationError(f"Seed {seed} decision CSV contains another seed")
        decisions.append(frame)
        metrics = pd.read_csv(metrics_path)
        if "seed" not in metrics or set(metrics["seed"].astype(int).unique()) != {seed}:
            raise VerificationError(f"Seed {seed} metrics CSV contains another seed")
        seed_output_summary[seed_key] = {
            "metrics_sha256": metrics_sha,
            "decisions_sha256": decision_sha,
            "evaluation_completion_sha256": completion_sha,
            "decision_rows": int(len(frame)),
        }

    aggregate_completion_path = output_root / "aggregate_complete.json"
    aggregate_completion_sha = _sha256(aggregate_completion_path)
    aggregate_completion = _load_json(aggregate_completion_path)
    _expect(
        aggregate_completion.get("schema_version"),
        AGGREGATE_COMPLETION_SCHEMA,
        "aggregate completion schema",
    )
    _expect(aggregate_completion.get("evaluation_fingerprint_sha256"), fingerprint, "aggregate fingerprint")
    _expect(aggregate_completion.get("evaluation_commitment_sha256"), commitment_sha, "aggregate actual commitment SHA-256")
    _expect(aggregate_completion.get("evaluation_started_sha256"), started_sha, "aggregate started SHA-256")
    _expect(aggregate_completion.get("seed_evaluation_completion_sha256"), seed_completion_hashes, "aggregate seed-completion DAG")

    aggregate_json_path = output_root / "aggregate_primary_analysis.json"
    aggregate_csv_path = output_root / "aggregate_primary_comparisons.csv"
    _expect(aggregate_completion.get("aggregate_json_sha256"), _sha256(aggregate_json_path), "aggregate JSON SHA-256")
    _expect(aggregate_completion.get("aggregate_csv_sha256"), _sha256(aggregate_csv_path), "aggregate CSV SHA-256")
    _expect(aggregate_completion.get("report_sha256"), _sha256(report_path), "primary report SHA-256")
    _expect(aggregate_completion.get("report_path"), primary_registration["report_path"], "aggregate report path")

    registry_completion = _load_json(registry_completion_path)
    _expect(registry_completion.get("schema_version"), REGISTRY_COMPLETION_SCHEMA, "registry completion schema")
    _expect(registry_completion.get("evaluation_fingerprint_sha256"), fingerprint, "registry fingerprint")
    _expect(registry_completion.get("evaluation_commitment_sha256"), commitment_sha, "registry actual commitment SHA-256")
    _expect(registry_completion.get("evaluation_started_sha256"), started_sha, "registry started SHA-256")
    _expect(Path(str(registry_completion.get("locked_output_root"))).resolve(), output_root, "registry output root")
    _expect(registry_completion.get("aggregate_completion_sha256"), aggregate_completion_sha, "registry aggregate-completion edge")
    _expect(registry_completion.get("status"), "prospective_test_evaluation_complete", "registry status")

    recomputed_analysis, recomputed_csv = recompute_primary_analysis(
        pd.concat(decisions, ignore_index=True), config
    )
    stored_analysis = _load_json(aggregate_json_path)
    _semantic_json_equal(stored_analysis, recomputed_analysis)
    recomputed_csv_text = recomputed_csv.to_csv(index=False, lineterminator="\n")
    if _normalized_text(aggregate_csv_path) != recomputed_csv_text:
        raise VerificationError("Stored aggregate CSV differs from recomputation")
    recomputed_report = render_primary_report(recomputed_analysis, recomputed_csv)
    if _normalized_text(report_path) != recomputed_report:
        raise VerificationError("Stored primary report differs from recomputation")
    _expect(
        aggregate_completion.get("primary_decision"),
        recomputed_analysis["decision"],
        "aggregate primary decision",
    )
    _expect(
        registry_completion.get("primary_decision"),
        recomputed_analysis["decision"],
        "registry primary decision",
    )

    return {
        "schema_version": VERIFICATION_SCHEMA,
        "verified": True,
        "claim_status": "consumed_development_exploratory",
        "primary_decision": recomputed_analysis["decision"],
        "verification_defect": newline_defect,
        "scoring_reexecuted": False,
        "model_inference_performed": False,
        "checkpoint_loaded": False,
        "prospective_npz_semantically_loaded": False,
        "saved_decisions_recomputed": True,
        "evaluation_fingerprint_sha256": fingerprint,
        "evaluation_commitment_actual_sha256": commitment_sha,
        "evaluation_commitment_canonical_lf_sha256": canonical_lf_sha,
        "evaluation_started_sha256": started_sha,
        "aggregate_completion_sha256": aggregate_completion_sha,
        "verified_seeds": list(EXPECTED_SEEDS),
        "seed_outputs": seed_output_summary,
        "declared_input_hashes_verified": [
            "evaluator_sources",
            "config",
            "protocol_summary_and_manifests",
            "fit_completions_and_artifacts",
            "embedding_caches_raw_sha256_only",
            "visibility_caches_raw_sha256_only",
        ],
        "declared_output_hashes_verified": [
            "five_metrics_csv",
            "five_decisions_csv",
            "five_evaluation_completions",
            "aggregate_json",
            "aggregate_csv",
            "primary_report",
            "aggregate_completion",
            "registry_completion",
        ],
        "primary_recomputation": {
            "implementation": "independent_post_evaluation_verifier",
            "dependency_components": recomputed_analysis["dependency_components"],
            "paired_bootstrap_resamples": recomputed_analysis["paired_bootstrap"][
                "resamples"
            ],
            "decision": recomputed_analysis["decision"],
            "aggregate_json_semantically_identical": True,
            "aggregate_csv_deterministically_identical": True,
            "primary_report_deterministically_identical": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--config", default="configs/cvian_sequence_utility_v1.json")
    parser.add_argument(
        "--protocol-dir", default="data/splits/ian_hurricane_sequence_four_role_v1"
    )
    parser.add_argument("--cache-root", default="outputs/cvian_sequence_active_v2/cache")
    parser.add_argument(
        "--output-root",
        default="outputs/cvian_sequence_active_v2/utility_experiment",
    )
    parser.add_argument(
        "--visibility-dir", default="outputs/cvian_sequence_active_v2/visibility"
    )
    parser.add_argument(
        "--registry-root",
        default="outputs/cvian_sequence_active_v2/evaluation_registry",
    )
    parser.add_argument(
        "--json-output",
        help="Optional new verification JSON path; existing files are never overwritten",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    result = verify_completion(
        repo_root=repo_root,
        config_path=_resolve(repo_root, args.config),
        protocol_dir=_resolve(repo_root, args.protocol_dir),
        cache_root=_resolve(repo_root, args.cache_root),
        output_root=_resolve(repo_root, args.output_root),
        visibility_dir=_resolve(repo_root, args.visibility_dir),
        registry_root=_resolve(repo_root, args.registry_root),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        output = _resolve(repo_root, args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
        except FileExistsError as exc:
            raise VerificationError(
                f"Refusing to overwrite verification output: {output}"
            ) from exc
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
