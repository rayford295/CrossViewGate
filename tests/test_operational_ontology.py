from __future__ import annotations

import copy
import json

import pytest

from crossview_conflict.decision.ontology import (
    DEFAULT_ONTOLOGY_PATH,
    OntologyUsageError,
    OntologyValidationError,
    build_cost_matrix,
    load_operational_ontology,
    validate_ontology,
)


def _raw_config() -> dict:
    return json.loads(DEFAULT_ONTOLOGY_PATH.read_text(encoding="utf-8"))


def test_preserves_native_labels_and_dataset_local_indices() -> None:
    ontology = load_operational_ontology()

    assert ontology.native_index_scope == "dataset_local"
    assert ontology.native_labels("eaton_wildfire") == (
        "No Damage",
        "Affected (1-9%)",
        "Minor (10-25%)",
        "Major (26-50%)",
        "Destroyed (>50%)",
        "Inaccessible",
    )
    assert ontology.native_labels("ian_hurricane") == (
        "0_MinorDamage",
        "1_ModerateDamage",
        "2_SevereDamage",
    )
    assert ontology.native_labels("milton_hurricane") == (
        "mild_damage",
        "moderate_damage",
        "severe_damage",
    )

    # Index zero has three different native meanings. Resolution requires the
    # dataset id and does not turn those names into semantic equivalents.
    assert ontology.label_from_index("eaton_wildfire", 0) == "No Damage"
    assert ontology.label_from_index("ian_hurricane", 0) == "0_MinorDamage"
    assert ontology.label_from_index("milton_hurricane", 0) == "mild_damage"


def test_explicit_action_mapping_does_not_replace_native_labels() -> None:
    ontology = load_operational_ontology()

    assert ontology.action_level("eaton_wildfire", "Destroyed (>50%)") == "urgent_response"
    assert ontology.action_level("ian_hurricane", "2_SevereDamage") == "urgent_response"
    assert ontology.action_level("milton_hurricane", "severe_damage") == "urgent_response"
    assert ontology.native_label("ian_hurricane", 2).name == "2_SevereDamage"


def test_wildfire_3class_is_read_only_derived_summary_with_provenance() -> None:
    ontology = load_operational_ontology()
    scheme = ontology.derived_scheme("eaton_wildfire", "wildfire_3class")

    assert scheme.provenance == "derived_summary"
    assert scheme.replaces_native_provenance is False
    assert ontology.derived_labels("eaton_wildfire", "wildfire_3class") == (
        "no_or_trace_damage",
        "damaged_repairable",
        "destroyed",
    )
    no_trace = ontology.to_derived_label(
        "eaton_wildfire", "wildfire_3class", "Affected (1-9%)"
    )
    assert no_trace is not None
    assert no_trace.name == "no_or_trace_damage"
    assert no_trace.source_native_labels == (
        "No Damage",
        "Affected (1-9%)",
    )
    assert ontology.derived_action_level(
        "eaton_wildfire", "wildfire_3class", "damaged_repairable"
    ) == "priority_inspection"

    # Derivation never mutates or replaces the exact six-class provenance.
    assert ontology.native_labels("eaton_wildfire")[-1] == "Inaccessible"
    assert ontology.to_derived_label(
        "eaton_wildfire", "wildfire_3class", "Inaccessible"
    ) is None
    with pytest.raises((AttributeError, TypeError)):
        scheme.labels += ()  # type: ignore[misc]

    matrix = ontology.derived_cost_matrix(
        "eaton_wildfire", "wildfire_3class"
    )
    assert matrix.labels == (
        "no_or_trace_damage",
        "damaged_repairable",
        "destroyed",
    )
    assert matrix.cost("destroyed", "no_or_trace_damage") == 8.0


def test_action_costs_are_directional_and_include_interventions() -> None:
    ontology = load_operational_ontology()

    assert ontology.error_type("routine_monitoring", "priority_inspection") == "adjacent_error"
    assert ontology.error_type("urgent_response", "routine_monitoring") == "severe_miss"
    assert ontology.error_type("routine_monitoring", "urgent_response") == "false_alarm"
    assert ontology.action_cost("urgent_response", "routine_monitoring") == 8.0
    assert ontology.action_cost("routine_monitoring", "urgent_response") == 3.0
    assert ontology.costs["extreme_error"] == 4.0
    assert ontology.intervention_cost("human_review") == 1.5
    assert ontology.intervention_cost("view_acquisition") == 0.5
    assert ontology.action_cost("urgent_response", "human_escalation") == 1.5


def test_cost_matrix_axes_and_native_decision_filter() -> None:
    ontology = load_operational_ontology()

    action_matrix = build_cost_matrix(ontology)
    assert action_matrix.labels == (
        "routine_monitoring",
        "priority_inspection",
        "urgent_response",
    )
    assert action_matrix.shape == (3, 3)
    assert action_matrix.cost("urgent_response", "routine_monitoring") == 8.0
    assert action_matrix.cost("routine_monitoring", "urgent_response") == 3.0

    eaton_matrix = ontology.cost_matrix("eaton_wildfire")
    assert eaton_matrix.shape == (5, 5)
    assert "Inaccessible" not in eaton_matrix.labels
    assert eaton_matrix.cost("No Damage", "Affected (1-9%)") == 0.0
    assert eaton_matrix.cost("Destroyed (>50%)", "No Damage") == 8.0


def test_inaccessible_is_access_constraint_not_uncertainty() -> None:
    ontology = load_operational_ontology()

    inaccessible = ontology.native_label("eaton_wildfire", "Inaccessible")
    assert inaccessible.action_level == "human_escalation"
    assert not inaccessible.decision_eligible
    ontology.validate_label_use("eaton_wildfire", "Inaccessible", "stress_test")
    ontology.validate_label_use("eaton_wildfire", "Inaccessible", "human_escalation")

    for forbidden_use in (
        "generic_uncertainty",
        "abstention_target",
        "primary_severity_class",
        "missing_view_marker",
    ):
        with pytest.raises(OntologyUsageError):
            ontology.validate_label_use(
                "eaton_wildfire", "Inaccessible", forbidden_use
            )

    with pytest.raises(OntologyUsageError):
        ontology.native_cost("eaton_wildfire", "Inaccessible", "No Damage")


def test_cross_event_metrics_fail_closed_and_macro_average_by_event() -> None:
    ontology = load_operational_ontology()

    assert ontology.is_cross_event_aggregatable("operational_expected_cost")
    assert ontology.is_cross_event_aggregatable("severe_miss_rate")
    assert not ontology.is_cross_event_aggregatable("native_macro_f1")
    assert ontology.aggregate_event_values(
        "operational_expected_cost",
        {
            "eaton_wildfire": 1.0,
            "ian_hurricane": 2.0,
            "milton_hurricane": 3.0,
        },
    ) == 2.0

    with pytest.raises(OntologyUsageError):
        ontology.validate_cross_event_aggregation(
            "native_confusion_matrix", ("eaton_wildfire", "ian_hurricane")
        )
    with pytest.raises(KeyError):
        ontology.is_cross_event_aggregatable("unregistered_metric")


def test_validation_rejects_index_equivalence_and_inaccessible_misuse() -> None:
    config = _raw_config()
    config["cross_event_aggregation"]["class_index_equivalence"] = "allowed"
    with pytest.raises(OntologyValidationError, match="class indices"):
        validate_ontology(config)

    config = copy.deepcopy(_raw_config())
    inaccessible = config["datasets"]["eaton_wildfire"]["native_labels"][-1]
    inaccessible["decision_eligible"] = True
    inaccessible["action_level"] = "urgent_response"
    with pytest.raises(OntologyValidationError, match="Inaccessible"):
        validate_ontology(config)


def test_validation_rejects_derived_scheme_that_discards_native_provenance() -> None:
    config = _raw_config()
    scheme = config["datasets"]["eaton_wildfire"]["derived_schemes"][
        "wildfire_3class"
    ]
    scheme["replaces_native_provenance"] = True
    with pytest.raises(OntologyValidationError, match="cannot replace"):
        validate_ontology(config)

    config = _raw_config()
    scheme = config["datasets"]["eaton_wildfire"]["derived_schemes"][
        "wildfire_3class"
    ]
    scheme["excluded_native_labels"] = []
    with pytest.raises(OntologyValidationError, match="unaccounted"):
        validate_ontology(config)


def test_loads_an_explicit_valid_config_path(tmp_path) -> None:
    path = tmp_path / "ontology.json"
    path.write_text(json.dumps(_raw_config()), encoding="utf-8")

    ontology = load_operational_ontology(path)

    assert ontology.schema_version == "1.0.0"
    assert ontology.ontology_id == "crossviewgate-operational-v1"
