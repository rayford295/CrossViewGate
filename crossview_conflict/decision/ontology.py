"""Validated operational label and cost ontology.

Native labels remain dataset scoped.  Cross-event comparisons are permitted only
after an explicit mapping to operational action levels; equal integer class
indices never establish label equivalence.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


DEFAULT_ONTOLOGY_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "operational_ontology.json"
)

REQUIRED_NATIVE_LABELS = {
    "eaton_wildfire": (
        "No Damage",
        "Affected (1-9%)",
        "Minor (10-25%)",
        "Major (26-50%)",
        "Destroyed (>50%)",
        "Inaccessible",
    ),
    "ian_hurricane": (
        "0_MinorDamage",
        "1_ModerateDamage",
        "2_SevereDamage",
    ),
    "milton_hurricane": (
        "mild_damage",
        "moderate_damage",
        "severe_damage",
    ),
}

REQUIRED_DERIVED_LABELS = {
    ("eaton_wildfire", "wildfire_3class"): (
        "no_or_trace_damage",
        "damaged_repairable",
        "destroyed",
    )
}

REQUIRED_COSTS = (
    "correct",
    "adjacent_error",
    "extreme_error",
    "severe_miss",
    "false_alarm",
    "human_review",
    "view_acquisition",
)


class OntologyValidationError(ValueError):
    """Raised when an ontology configuration violates the policy contract."""


class OntologyUsageError(ValueError):
    """Raised when a valid label is used outside its permitted semantics."""


@dataclass(frozen=True)
class ActionLevel:
    """A dataset-independent operational action."""

    id: str
    rank: int | None
    decision_eligible: bool
    description: str


@dataclass(frozen=True)
class NativeLabel:
    """A label whose integer index is meaningful only inside one dataset."""

    name: str
    native_index: int
    action_level: str
    decision_eligible: bool
    semantics: str = ""


@dataclass(frozen=True)
class DerivedLabel:
    """A read-only summary label with explicit native-label provenance."""

    name: str
    derived_index: int
    source_native_labels: tuple[str, ...]
    action_level: str


@dataclass(frozen=True)
class DerivedScheme:
    """A derived reporting view that cannot replace source-native labels."""

    id: str
    provenance: str
    replaces_native_provenance: bool
    labels: tuple[DerivedLabel, ...]
    excluded_native_labels: tuple[str, ...]

    def resolve(self, label_or_index: str | int) -> DerivedLabel:
        if isinstance(label_or_index, bool):
            raise KeyError(f"Boolean is not a derived index for {self.id!r}.")
        if isinstance(label_or_index, int):
            for label in self.labels:
                if label.derived_index == label_or_index:
                    return label
            raise KeyError(f"Unknown derived index {label_or_index} for {self.id!r}.")
        for label in self.labels:
            if label.name == label_or_index:
                return label
        raise KeyError(f"Unknown derived label {label_or_index!r} for {self.id!r}.")


@dataclass(frozen=True)
class DatasetOntology:
    """Native label contract for one event dataset."""

    id: str
    event: str
    hazard: str
    labels: tuple[NativeLabel, ...]
    derived_schemes: tuple[DerivedScheme, ...] = ()

    def resolve(self, label_or_index: str | int) -> NativeLabel:
        """Resolve an exact native name or a dataset-local integer index."""

        if isinstance(label_or_index, bool):
            raise KeyError(f"Boolean is not a native index for {self.id!r}.")
        if isinstance(label_or_index, int):
            for label in self.labels:
                if label.native_index == label_or_index:
                    return label
            raise KeyError(
                f"Unknown dataset-local index {label_or_index} for {self.id!r}."
            )
        for label in self.labels:
            if label.name == label_or_index:
                return label
        raise KeyError(f"Unknown native label {label_or_index!r} for {self.id!r}.")

    def derived_scheme(self, scheme_id: str) -> DerivedScheme:
        for scheme in self.derived_schemes:
            if scheme.id == scheme_id:
                return scheme
        raise KeyError(f"Unknown derived scheme {scheme_id!r} for {self.id!r}.")


@dataclass(frozen=True)
class AggregationRule:
    """Whether and how an event-level metric can be combined across events."""

    metric: str
    scope: str
    reducer: str | None
    basis: str

    @property
    def cross_event(self) -> bool:
        return self.scope == "cross_event"


@dataclass(frozen=True)
class CostMatrix:
    """Immutable cost matrix with actual labels on rows and predictions on columns."""

    labels: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.values), len(self.labels))

    @property
    def row_labels(self) -> tuple[str, ...]:
        return self.labels

    @property
    def column_labels(self) -> tuple[str, ...]:
        return self.labels

    def __getitem__(self, key: int | tuple[int, int]) -> float | tuple[float, ...]:
        if isinstance(key, tuple):
            row, column = key
            return self.values[row][column]
        return self.values[key]

    def cost(self, actual: str, predicted: str) -> float:
        try:
            row = self.labels.index(actual)
            column = self.labels.index(predicted)
        except ValueError as exc:
            raise KeyError(f"Label is not present in this cost matrix: {exc}") from exc
        return self.values[row][column]

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self.values]

    def as_numpy(self):
        """Return a defensive NumPy copy without making NumPy an import-time need."""

        import numpy as np

        return np.asarray(self.values, dtype=float).copy()


class OperationalOntology:
    """Read-only, validated ontology used by training and evaluation code."""

    def __init__(
        self,
        *,
        schema_version: str,
        ontology_id: str,
        action_levels: tuple[ActionLevel, ...],
        costs: Mapping[str, float],
        datasets: Mapping[str, DatasetOntology],
        special_labels: Mapping[str, Mapping[str, Any]],
        aggregation_rules: Mapping[str, AggregationRule],
    ) -> None:
        self.schema_version = schema_version
        self.ontology_id = ontology_id
        self.action_levels = action_levels
        self.costs = MappingProxyType(dict(costs))
        self.datasets = MappingProxyType(dict(datasets))
        self.special_labels = MappingProxyType(
            {
                key: MappingProxyType(dict(value))
                for key, value in special_labels.items()
            }
        )
        self.aggregation_rules = MappingProxyType(dict(aggregation_rules))
        self.native_index_scope = "dataset_local"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "OperationalOntology":
        validate_ontology(config)

        action_levels = tuple(
            ActionLevel(
                id=item["id"],
                rank=item["rank"],
                decision_eligible=item["decision_eligible"],
                description=item["description"],
            )
            for item in config["action_levels"]
        )
        datasets = {
            dataset_id: DatasetOntology(
                id=dataset_id,
                event=item["event"],
                hazard=item["hazard"],
                labels=tuple(
                    NativeLabel(
                        name=label["name"],
                        native_index=label["native_index"],
                        action_level=label["action_level"],
                        decision_eligible=label["decision_eligible"],
                        semantics=label.get("semantics", ""),
                    )
                    for label in sorted(
                        item["native_labels"], key=lambda value: value["native_index"]
                    )
                ),
                derived_schemes=tuple(
                    DerivedScheme(
                        id=scheme_id,
                        provenance=scheme["provenance"],
                        replaces_native_provenance=scheme[
                            "replaces_native_provenance"
                        ],
                        labels=tuple(
                            DerivedLabel(
                                name=label["name"],
                                derived_index=label["derived_index"],
                                source_native_labels=tuple(
                                    label["source_native_labels"]
                                ),
                                action_level=label["action_level"],
                            )
                            for label in sorted(
                                scheme["labels"],
                                key=lambda value: value["derived_index"],
                            )
                        ),
                        excluded_native_labels=tuple(
                            scheme["excluded_native_labels"]
                        ),
                    )
                    for scheme_id, scheme in item.get(
                        "derived_schemes", {}
                    ).items()
                ),
            )
            for dataset_id, item in config["datasets"].items()
        }
        aggregation_rules = {
            metric: AggregationRule(
                metric=metric,
                scope=rule["scope"],
                reducer=rule["reducer"],
                basis=rule["basis"],
            )
            for metric, rule in config["cross_event_aggregation"]["metrics"].items()
        }
        return cls(
            schema_version=config["schema_version"],
            ontology_id=config["ontology_id"],
            action_levels=action_levels,
            costs={key: float(value) for key, value in config["costs"].items()},
            datasets=datasets,
            special_labels=config["special_labels"],
            aggregation_rules=aggregation_rules,
        )

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(self.datasets)

    @property
    def decision_action_ids(self) -> tuple[str, ...]:
        return tuple(
            action.id
            for action in sorted(
                (item for item in self.action_levels if item.decision_eligible),
                key=lambda item: item.rank if item.rank is not None else -1,
            )
        )

    def dataset(self, dataset_id: str) -> DatasetOntology:
        try:
            return self.datasets[dataset_id]
        except KeyError as exc:
            raise KeyError(f"Unknown ontology dataset: {dataset_id!r}") from exc

    def native_label(
        self, dataset_id: str, label_or_index: str | int
    ) -> NativeLabel:
        return self.dataset(dataset_id).resolve(label_or_index)

    def native_labels(
        self, dataset_id: str, *, decision_only: bool = False
    ) -> tuple[str, ...]:
        labels = self.dataset(dataset_id).labels
        if decision_only:
            labels = tuple(label for label in labels if label.decision_eligible)
        return tuple(label.name for label in labels)

    def native_index(self, dataset_id: str, native_label: str) -> int:
        return self.native_label(dataset_id, native_label).native_index

    def label_from_index(self, dataset_id: str, native_index: int) -> str:
        """Resolve an index only when its dataset is explicitly supplied."""

        return self.native_label(dataset_id, native_index).name

    def action_level(self, dataset_id: str, label_or_index: str | int) -> str:
        return self.native_label(dataset_id, label_or_index).action_level

    def derived_scheme(self, dataset_id: str, scheme_id: str) -> DerivedScheme:
        """Return a frozen derived view while preserving the native dataset."""

        return self.dataset(dataset_id).derived_scheme(scheme_id)

    def derived_labels(self, dataset_id: str, scheme_id: str) -> tuple[str, ...]:
        return tuple(
            label.name for label in self.derived_scheme(dataset_id, scheme_id).labels
        )

    def derived_action_level(
        self, dataset_id: str, scheme_id: str, label_or_index: str | int
    ) -> str:
        return self.derived_scheme(dataset_id, scheme_id).resolve(
            label_or_index
        ).action_level

    def to_derived_label(
        self,
        dataset_id: str,
        scheme_id: str,
        native_label_or_index: str | int,
    ) -> DerivedLabel | None:
        """Map a native label to a derived summary without discarding provenance.

        ``None`` means the source label is explicitly excluded from the scheme.
        The returned object retains all native labels that make up its group.
        """

        native = self.native_label(dataset_id, native_label_or_index)
        scheme = self.derived_scheme(dataset_id, scheme_id)
        if native.name in scheme.excluded_native_labels:
            return None
        for derived in scheme.labels:
            if native.name in derived.source_native_labels:
                return derived
        raise OntologyUsageError(
            f"Native label {native.name!r} has no declared provenance path in "
            f"derived scheme {scheme_id!r}."
        )

    def get_action_level(self, action_id: str) -> ActionLevel:
        for action in self.action_levels:
            if action.id == action_id:
                return action
        raise KeyError(f"Unknown action level: {action_id!r}")

    def error_type(self, actual_action: str, predicted_action: str) -> str:
        """Classify an action error before looking up its policy cost."""

        actual = self.get_action_level(actual_action)
        predicted = self.get_action_level(predicted_action)
        if not actual.decision_eligible:
            raise OntologyUsageError(
                f"{actual.id!r} is an escalation state, not a severity truth."
            )
        if not predicted.decision_eligible:
            if predicted.id == "human_escalation":
                return "human_review"
            raise OntologyUsageError(
                f"{predicted.id!r} has no operational decision cost."
            )
        if actual.id == predicted.id:
            return "correct"

        assert actual.rank is not None and predicted.rank is not None
        distance = abs(actual.rank - predicted.rank)
        if distance == 1:
            return "adjacent_error"

        ranks = [
            action.rank
            for action in self.action_levels
            if action.decision_eligible and action.rank is not None
        ]
        if actual.rank == max(ranks) and predicted.rank == min(ranks):
            return "severe_miss"
        if actual.rank == min(ranks) and predicted.rank == max(ranks):
            return "false_alarm"
        return "extreme_error"

    def action_cost(self, actual_action: str, predicted_action: str) -> float:
        return self.costs[self.error_type(actual_action, predicted_action)]

    def intervention_cost(self, intervention: str) -> float:
        if intervention not in {"human_review", "view_acquisition"}:
            raise KeyError(f"Unknown intervention cost: {intervention!r}")
        return self.costs[intervention]

    def native_cost(
        self,
        dataset_id: str,
        actual_label: str | int,
        predicted_label: str | int,
    ) -> float:
        """Return policy cost without treating native labels as interchangeable."""

        actual = self.native_label(dataset_id, actual_label)
        predicted = self.native_label(dataset_id, predicted_label)
        if not actual.decision_eligible or not predicted.decision_eligible:
            raise OntologyUsageError(
                "Non-decision labels such as Inaccessible are excluded from "
                "severity cost matrices; use explicit human escalation instead."
            )
        return self.action_cost(actual.action_level, predicted.action_level)

    cost = native_cost

    def action_cost_matrix(self) -> CostMatrix:
        labels = self.decision_action_ids
        values = tuple(
            tuple(self.action_cost(actual, predicted) for predicted in labels)
            for actual in labels
        )
        return CostMatrix(labels=labels, values=values)

    def cost_matrix(self, dataset_id: str | None = None) -> CostMatrix:
        """Build the action matrix or a dataset-native decision-label matrix."""

        if dataset_id is None:
            return self.action_cost_matrix()
        labels = tuple(
            label
            for label in self.dataset(dataset_id).labels
            if label.decision_eligible
        )
        values = tuple(
            tuple(
                self.action_cost(actual.action_level, predicted.action_level)
                for predicted in labels
            )
            for actual in labels
        )
        return CostMatrix(
            labels=tuple(label.name for label in labels),
            values=values,
        )

    def derived_cost_matrix(self, dataset_id: str, scheme_id: str) -> CostMatrix:
        """Build a matrix for an explicitly declared derived reporting scheme."""

        scheme = self.derived_scheme(dataset_id, scheme_id)
        labels = scheme.labels
        values = tuple(
            tuple(
                self.action_cost(actual.action_level, predicted.action_level)
                for predicted in labels
            )
            for actual in labels
        )
        return CostMatrix(
            labels=tuple(label.name for label in labels),
            values=values,
        )

    build_cost_matrix = cost_matrix

    @property
    def inaccessible_policy(self) -> Mapping[str, Any]:
        return self.special_labels["inaccessible"]

    def validate_label_use(
        self, dataset_id: str, label_or_index: str | int, use: str
    ) -> None:
        """Fail closed when ``Inaccessible`` is used as generic uncertainty."""

        label = self.native_label(dataset_id, label_or_index)
        policy = self.inaccessible_policy
        if dataset_id != policy["dataset"] or label.name != policy["native_label"]:
            return
        if use not in policy["allowed_uses"]:
            raise OntologyUsageError(
                "Inaccessible is an inspector-recorded field-access constraint. "
                f"Use {use!r} is not allowed; permitted uses are "
                f"{tuple(policy['allowed_uses'])}."
            )

    def is_label_use_allowed(
        self, dataset_id: str, label_or_index: str | int, use: str
    ) -> bool:
        try:
            self.validate_label_use(dataset_id, label_or_index, use)
        except OntologyUsageError:
            return False
        return True

    def aggregation_rule(self, metric: str) -> AggregationRule:
        try:
            return self.aggregation_rules[metric]
        except KeyError as exc:
            raise KeyError(
                f"Metric {metric!r} has no aggregation rule; cross-event "
                "aggregation fails closed."
            ) from exc

    def is_cross_event_aggregatable(self, metric: str) -> bool:
        return self.aggregation_rule(metric).cross_event

    can_aggregate_cross_event = is_cross_event_aggregatable

    def validate_cross_event_aggregation(
        self, metric: str, dataset_ids: Sequence[str]
    ) -> AggregationRule:
        rule = self.aggregation_rule(metric)
        unique_ids = tuple(dict.fromkeys(dataset_ids))
        if len(unique_ids) < 2:
            raise OntologyUsageError(
                "Cross-event aggregation requires at least two distinct datasets."
            )
        for dataset_id in unique_ids:
            self.dataset(dataset_id)
        if not rule.cross_event:
            raise OntologyUsageError(
                f"{metric!r} is event-only because it is defined on native labels."
            )
        return rule

    def aggregate_event_values(
        self, metric: str, values_by_dataset: Mapping[str, float]
    ) -> float:
        """Macro-average an allowed metric while preserving event breakdown."""

        rule = self.validate_cross_event_aggregation(
            metric, tuple(values_by_dataset)
        )
        if rule.reducer != "event_macro":
            raise OntologyUsageError(
                f"Unsupported configured reducer for {metric!r}: {rule.reducer!r}."
            )
        values = [float(value) for value in values_by_dataset.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Event metrics must be finite numbers.")
        return sum(values) / len(values)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OntologyValidationError(f"{location} must be an object.")
    return value


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], location: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise OntologyValidationError(f"{location} is missing keys: {missing}.")


def _validate_actions(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_actions = config["action_levels"]
    if not _is_sequence(raw_actions) or not raw_actions:
        raise OntologyValidationError("action_levels must be a non-empty array.")
    actions: dict[str, Mapping[str, Any]] = {}
    ranks: list[int] = []
    for index, raw_action in enumerate(raw_actions):
        action = _require_mapping(raw_action, f"action_levels[{index}]")
        _require_keys(
            action,
            ("id", "rank", "decision_eligible", "description"),
            f"action_levels[{index}]",
        )
        action_id = action["id"]
        if not isinstance(action_id, str) or not action_id:
            raise OntologyValidationError("Every action id must be a non-empty string.")
        if action_id in actions:
            raise OntologyValidationError(f"Duplicate action id: {action_id!r}.")
        if not isinstance(action["decision_eligible"], bool):
            raise OntologyValidationError(
                f"decision_eligible for {action_id!r} must be boolean."
            )
        rank = action["rank"]
        if action["decision_eligible"]:
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise OntologyValidationError(
                    f"Decision action {action_id!r} needs a non-negative integer rank."
                )
            ranks.append(rank)
        elif rank is not None:
            raise OntologyValidationError(
                f"Non-decision action {action_id!r} must have a null rank."
            )
        actions[action_id] = action
    if sorted(ranks) != list(range(len(ranks))):
        raise OntologyValidationError(
            "Decision action ranks must be unique and contiguous from zero."
        )
    if "human_escalation" not in actions or actions["human_escalation"][
        "decision_eligible"
    ]:
        raise OntologyValidationError(
            "human_escalation must exist as a non-decision action."
        )
    if len(ranks) < 3:
        raise OntologyValidationError(
            "At least three ordinal decision actions are required to define "
            "adjacent and extreme errors."
        )
    return actions


def _validate_costs(config: Mapping[str, Any]) -> None:
    costs = _require_mapping(config["costs"], "costs")
    _require_keys(costs, REQUIRED_COSTS, "costs")
    for key in REQUIRED_COSTS:
        value = costs[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OntologyValidationError(f"Cost {key!r} must be numeric.")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise OntologyValidationError(
                f"Cost {key!r} must be finite and non-negative."
            )
    if float(costs["correct"]) != 0:
        raise OntologyValidationError("The correct-decision cost must be zero.")
    if float(costs["extreme_error"]) <= float(costs["adjacent_error"]):
        raise OntologyValidationError(
            "extreme_error must cost more than adjacent_error."
        )
    if float(costs["severe_miss"]) <= float(costs["extreme_error"]):
        raise OntologyValidationError(
            "severe_miss must cost more than a generic extreme_error."
        )


def _validate_derived_schemes(
    dataset_id: str,
    dataset: Mapping[str, Any],
    actions: Mapping[str, Mapping[str, Any]],
    native_by_name: Mapping[str, Mapping[str, Any]],
) -> None:
    raw_schemes = _require_mapping(
        dataset.get("derived_schemes", {}),
        f"datasets.{dataset_id}.derived_schemes",
    )
    required_for_dataset = {
        scheme_id
        for required_dataset, scheme_id in REQUIRED_DERIVED_LABELS
        if required_dataset == dataset_id
    }
    missing = sorted(required_for_dataset - set(raw_schemes))
    if missing:
        raise OntologyValidationError(
            f"Required derived schemes for {dataset_id!r} are missing: {missing}."
        )

    for scheme_id, raw_scheme in raw_schemes.items():
        location = f"datasets.{dataset_id}.derived_schemes.{scheme_id}"
        scheme = _require_mapping(raw_scheme, location)
        _require_keys(
            scheme,
            (
                "provenance",
                "replaces_native_provenance",
                "labels",
                "excluded_native_labels",
            ),
            location,
        )
        if scheme["provenance"] != "derived_summary":
            raise OntologyValidationError(
                f"{location} must identify itself as a derived_summary."
            )
        if scheme["replaces_native_provenance"] is not False:
            raise OntologyValidationError(
                f"{location} cannot replace source-native provenance."
            )
        raw_labels = scheme["labels"]
        excluded = scheme["excluded_native_labels"]
        if not _is_sequence(raw_labels) or not raw_labels:
            raise OntologyValidationError(f"{location}.labels must be non-empty.")
        if not _is_sequence(excluded):
            raise OntologyValidationError(
                f"{location}.excluded_native_labels must be an array."
            )
        if any(not isinstance(name, str) for name in excluded):
            raise OntologyValidationError(
                f"{location}.excluded_native_labels must contain native names."
            )
        if len(set(excluded)) != len(excluded):
            raise OntologyValidationError(
                f"{location}.excluded_native_labels contains duplicates."
            )
        unknown_excluded = sorted(set(excluded) - set(native_by_name))
        if unknown_excluded:
            raise OntologyValidationError(
                f"{location} excludes unknown native labels: {unknown_excluded}."
            )

        derived_names: set[str] = set()
        derived_indices: set[int] = set()
        labels_by_index: dict[int, str] = {}
        assigned_sources: set[str] = set()
        for index, raw_label in enumerate(raw_labels):
            label_location = f"{location}.labels[{index}]"
            label = _require_mapping(raw_label, label_location)
            _require_keys(
                label,
                (
                    "name",
                    "derived_index",
                    "source_native_labels",
                    "action_level",
                ),
                label_location,
            )
            name = label["name"]
            derived_index = label["derived_index"]
            sources = label["source_native_labels"]
            action_id = label["action_level"]
            if not isinstance(name, str) or not name:
                raise OntologyValidationError(
                    f"{label_location}.name must be non-empty."
                )
            if name in derived_names:
                raise OntologyValidationError(
                    f"Duplicate derived label {name!r} in {location}."
                )
            if (
                isinstance(derived_index, bool)
                or not isinstance(derived_index, int)
                or derived_index < 0
            ):
                raise OntologyValidationError(
                    f"{label_location}.derived_index must be non-negative."
                )
            if derived_index in derived_indices:
                raise OntologyValidationError(
                    f"Duplicate derived index {derived_index} in {location}."
                )
            if not _is_sequence(sources) or not sources:
                raise OntologyValidationError(
                    f"{label_location}.source_native_labels must be non-empty."
                )
            if any(not isinstance(source, str) for source in sources):
                raise OntologyValidationError(
                    f"{label_location}.source_native_labels must contain names."
                )
            if len(set(sources)) != len(sources):
                raise OntologyValidationError(
                    f"{label_location}.source_native_labels contains duplicates."
                )
            unknown_sources = sorted(set(sources) - set(native_by_name))
            if unknown_sources:
                raise OntologyValidationError(
                    f"{label_location} references unknown native labels: "
                    f"{unknown_sources}."
                )
            overlap = sorted(set(sources) & assigned_sources)
            if overlap:
                raise OntologyValidationError(
                    f"Native labels cannot appear in multiple derived groups: {overlap}."
                )
            if action_id not in actions or not actions[action_id]["decision_eligible"]:
                raise OntologyValidationError(
                    f"Derived label {name!r} needs a decision-eligible action."
                )
            mismatched = [
                source
                for source in sources
                if native_by_name[source]["action_level"] != action_id
            ]
            if mismatched:
                raise OntologyValidationError(
                    f"Derived label {name!r} changes the action mapping of source "
                    f"labels: {mismatched}."
                )
            derived_names.add(name)
            derived_indices.add(derived_index)
            labels_by_index[derived_index] = name
            assigned_sources.update(sources)

        if sorted(derived_indices) != list(range(len(derived_indices))):
            raise OntologyValidationError(
                f"Derived indices in {location} must be contiguous from zero."
            )
        if assigned_sources & set(excluded):
            raise OntologyValidationError(
                f"{location} cannot both derive and exclude a native label."
            )
        if assigned_sources | set(excluded) != set(native_by_name):
            unaccounted = sorted(
                set(native_by_name) - assigned_sources - set(excluded)
            )
            raise OntologyValidationError(
                f"{location} leaves native provenance unaccounted for: {unaccounted}."
            )
        required_names = REQUIRED_DERIVED_LABELS.get((dataset_id, scheme_id))
        if required_names is not None:
            actual_names = tuple(
                labels_by_index[index] for index in sorted(labels_by_index)
            )
            if actual_names != required_names:
                raise OntologyValidationError(
                    f"Derived labels for {dataset_id}.{scheme_id} changed. "
                    f"Expected {required_names}, received {actual_names}."
                )
            if tuple(excluded) != ("Inaccessible",):
                raise OntologyValidationError(
                    "wildfire_3class must exclude Inaccessible and retain it only "
                    "in source-native provenance."
                )


def _validate_datasets(
    config: Mapping[str, Any], actions: Mapping[str, Mapping[str, Any]]
) -> None:
    datasets = _require_mapping(config["datasets"], "datasets")
    missing_datasets = sorted(set(REQUIRED_NATIVE_LABELS) - set(datasets))
    if missing_datasets:
        raise OntologyValidationError(
            f"Required datasets are missing: {missing_datasets}."
        )
    for dataset_id, raw_dataset in datasets.items():
        dataset = _require_mapping(raw_dataset, f"datasets.{dataset_id}")
        _require_keys(dataset, ("event", "hazard", "native_labels"), f"datasets.{dataset_id}")
        raw_labels = dataset["native_labels"]
        if not _is_sequence(raw_labels) or not raw_labels:
            raise OntologyValidationError(
                f"datasets.{dataset_id}.native_labels must be a non-empty array."
            )
        names: set[str] = set()
        indices: set[int] = set()
        labels_by_index: dict[int, str] = {}
        native_by_name: dict[str, Mapping[str, Any]] = {}
        for index, raw_label in enumerate(raw_labels):
            location = f"datasets.{dataset_id}.native_labels[{index}]"
            label = _require_mapping(raw_label, location)
            _require_keys(
                label,
                ("name", "native_index", "action_level", "decision_eligible"),
                location,
            )
            name = label["name"]
            native_index = label["native_index"]
            action_id = label["action_level"]
            if not isinstance(name, str) or not name:
                raise OntologyValidationError(f"{location}.name must be non-empty.")
            if name in names:
                raise OntologyValidationError(
                    f"Duplicate native label {name!r} in {dataset_id!r}."
                )
            if isinstance(native_index, bool) or not isinstance(native_index, int):
                raise OntologyValidationError(
                    f"Native index for {name!r} must be an integer."
                )
            if native_index in indices:
                raise OntologyValidationError(
                    f"Duplicate native index {native_index} in {dataset_id!r}."
                )
            if action_id not in actions:
                raise OntologyValidationError(
                    f"Unknown action {action_id!r} for {dataset_id}.{name}."
                )
            if not isinstance(label["decision_eligible"], bool):
                raise OntologyValidationError(
                    f"decision_eligible for {dataset_id}.{name} must be boolean."
                )
            if label["decision_eligible"] != actions[action_id]["decision_eligible"]:
                raise OntologyValidationError(
                    f"Decision eligibility for {dataset_id}.{name} conflicts with "
                    f"action {action_id!r}."
                )
            names.add(name)
            indices.add(native_index)
            labels_by_index[native_index] = name
            native_by_name[name] = label
        if sorted(indices) != list(range(len(indices))):
            raise OntologyValidationError(
                f"Native indices for {dataset_id!r} must be contiguous from zero."
            )
        if dataset_id in REQUIRED_NATIVE_LABELS:
            actual = tuple(labels_by_index[index] for index in sorted(labels_by_index))
            expected = REQUIRED_NATIVE_LABELS[dataset_id]
            if actual != expected:
                raise OntologyValidationError(
                    f"Native labels for {dataset_id!r} changed. Expected {expected}, "
                    f"received {actual}."
                )
        _validate_derived_schemes(
            dataset_id,
            dataset,
            actions,
            native_by_name,
        )


def _validate_inaccessible(config: Mapping[str, Any]) -> None:
    special_labels = _require_mapping(config["special_labels"], "special_labels")
    policy = _require_mapping(
        special_labels.get("inaccessible"), "special_labels.inaccessible"
    )
    _require_keys(
        policy,
        (
            "dataset",
            "native_label",
            "semantic_type",
            "action_level",
            "allowed_uses",
            "forbidden_uses",
        ),
        "special_labels.inaccessible",
    )
    if (
        policy["dataset"] != "eaton_wildfire"
        or policy["native_label"] != "Inaccessible"
        or policy["semantic_type"] != "field_access_constraint"
        or policy["action_level"] != "human_escalation"
    ):
        raise OntologyValidationError(
            "Inaccessible must denote the Eaton field-access constraint and map "
            "only to human_escalation."
        )
    allowed = policy["allowed_uses"]
    forbidden = policy["forbidden_uses"]
    if not _is_sequence(allowed) or not _is_sequence(forbidden):
        raise OntologyValidationError(
            "Inaccessible allowed_uses and forbidden_uses must be arrays."
        )
    if not {"stress_test", "human_escalation"}.issubset(set(allowed)):
        raise OntologyValidationError(
            "Inaccessible is allowed only as a stress-test stratum or human escalation."
        )
    if not {"generic_uncertainty", "abstention_target"}.issubset(set(forbidden)):
        raise OntologyValidationError(
            "Inaccessible must explicitly forbid generic uncertainty and abstention use."
        )
    if set(allowed) & set(forbidden):
        raise OntologyValidationError(
            "Inaccessible allowed and forbidden uses must not overlap."
        )
    eaton_labels = config["datasets"]["eaton_wildfire"]["native_labels"]
    inaccessible = next(
        label for label in eaton_labels if label["name"] == "Inaccessible"
    )
    if inaccessible["decision_eligible"] or inaccessible["action_level"] != "human_escalation":
        raise OntologyValidationError(
            "Inaccessible cannot be a decision-eligible severity class."
        )


def _validate_aggregation(config: Mapping[str, Any]) -> None:
    aggregation = _require_mapping(
        config["cross_event_aggregation"], "cross_event_aggregation"
    )
    _require_keys(
        aggregation,
        (
            "native_label_join",
            "class_index_equivalence",
            "allowed_basis",
            "default_reducer",
            "requires_event_breakdown",
            "metrics",
        ),
        "cross_event_aggregation",
    )
    if aggregation["native_label_join"] != "forbidden":
        raise OntologyValidationError("Cross-event native-label joins must be forbidden.")
    if aggregation["class_index_equivalence"] != "forbidden":
        raise OntologyValidationError(
            "Equal class indices must not imply cross-dataset equivalence."
        )
    if aggregation["allowed_basis"] != "explicit_action_level":
        raise OntologyValidationError(
            "Cross-event aggregation requires explicit action-level mappings."
        )
    if aggregation["default_reducer"] != "event_macro":
        raise OntologyValidationError(
            "Cross-event results must be macro-averaged by event."
        )
    if aggregation["requires_event_breakdown"] is not True:
        raise OntologyValidationError(
            "Cross-event aggregates must retain an event-level breakdown."
        )
    metrics = _require_mapping(aggregation["metrics"], "cross_event_aggregation.metrics")
    required_cross_event = {
        "operational_expected_cost",
        "severe_miss_rate",
        "false_alarm_rate",
        "human_review_rate",
        "view_acquisition_rate",
    }
    required_event_only = {
        "native_confusion_matrix",
        "native_macro_f1",
        "native_class_recall",
    }
    missing = sorted((required_cross_event | required_event_only) - set(metrics))
    if missing:
        raise OntologyValidationError(f"Aggregation rules are missing: {missing}.")
    for metric, raw_rule in metrics.items():
        rule = _require_mapping(raw_rule, f"aggregation metric {metric}")
        _require_keys(rule, ("scope", "reducer", "basis"), f"aggregation metric {metric}")
        if rule["scope"] not in {"cross_event", "event_only"}:
            raise OntologyValidationError(
                f"Invalid aggregation scope for {metric!r}: {rule['scope']!r}."
            )
        if metric in required_cross_event and (
            rule["scope"] != "cross_event" or rule["reducer"] != "event_macro"
        ):
            raise OntologyValidationError(
                f"{metric!r} must use cross-event event-macro aggregation."
            )
        if metric in required_event_only and (
            rule["scope"] != "event_only" or rule["reducer"] is not None
        ):
            raise OntologyValidationError(f"{metric!r} must remain event-only.")


def validate_ontology(config: Mapping[str, Any]) -> None:
    """Validate structure plus non-negotiable operational semantics."""

    config = _require_mapping(config, "ontology")
    _require_keys(
        config,
        (
            "schema_version",
            "ontology_id",
            "native_index_scope",
            "action_levels",
            "costs",
            "datasets",
            "special_labels",
            "cross_event_aggregation",
        ),
        "ontology",
    )
    if not isinstance(config["schema_version"], str) or not config["schema_version"]:
        raise OntologyValidationError("schema_version must be a non-empty string.")
    if not isinstance(config["ontology_id"], str) or not config["ontology_id"]:
        raise OntologyValidationError("ontology_id must be a non-empty string.")
    if config["native_index_scope"] != "dataset_local":
        raise OntologyValidationError("Native class indices must be dataset-local.")
    actions = _validate_actions(config)
    _validate_costs(config)
    _validate_datasets(config, actions)
    _validate_inaccessible(config)
    _validate_aggregation(config)


def load_operational_ontology(
    path: str | Path | None = None,
) -> OperationalOntology:
    """Load and validate an ontology JSON file.

    The repository configuration is used when ``path`` is omitted.
    """

    resolved = DEFAULT_ONTOLOGY_PATH if path is None else Path(path).expanduser()
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as exc:
        raise OntologyValidationError(
            f"Invalid JSON in operational ontology {resolved}: {exc}"
        ) from exc
    return OperationalOntology.from_mapping(config)


load_ontology = load_operational_ontology


def build_cost_matrix(
    ontology: OperationalOntology, dataset_id: str | None = None
) -> CostMatrix:
    """Functional wrapper for callers that do not retain bound methods."""

    return ontology.cost_matrix(dataset_id)
