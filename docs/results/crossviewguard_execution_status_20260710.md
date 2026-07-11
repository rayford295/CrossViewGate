# CrossViewGuard Current-Data Execution Status

> **Focused Eaton update (2026-07-11):** the repository now has a frozen
> four-role Eaton spatial protocol and fresh role-isolated model execution, so
> the older statements below that predictions and spatial roles are absent are
> superseded. The focused v1 branch nevertheless stops before annotation:
> `study_development` has only 28 total spatial blocks and cannot satisfy the
> registered minimum of 30 eligible blocks. The completed five-seed ensemble
> likewise yields only 97 unsigned disagreements across 24 blocks, below the
> 250/30 registered capacity gates. See
> [`eaton_component_observability_v1.md`](eaton_component_observability_v1.md).

- **Protocol date:** 2026-07-10
- **Execution closure:** 2026-07-11
- **Final status:** **current-data executable branches reached their endpoint;
  the central CrossViewGuard hypothesis is incomplete and unconfirmed**
- **Scope:** CVIAN label-cost active view, Milton frozen zero-shot sensitivity,
  RQ1 disagreement anatomy, RQ2/RQ3 attestation dependency gate, and the
  downstream RQ4 risk-control prerequisite

## Executive decision

The current repositories and local data no longer have a legitimate unexecuted
experiment that can confirm the central CrossViewGuard claim. This is an
execution endpoint for the **current-data branch**, not completion or
falsification of the research idea.

The central claim has two linked parts:

1. cross-view disagreement encodes view-conditional attestability; and
2. acquiring evidence to complete attestation coverage improves decisions under
   a fixed evidence/risk budget.

Neither part is confirmed. CVIAN's label-cost selector is NO-GO, Milton supplies
only a bounded sensitivity signal, RQ1 met 0/3 development criteria, and the
annotations required to execute RQ2/RQ3 do not exist. The next scientific action
is new external data/annotation acquisition followed by a newly frozen,
role-isolated protocol. Further tuning on consumed CVIAN or Milton v1 artifacts
is not a valid continuation.

## Decision table

| Branch | Execution status | Decision / claim boundary |
|---|---|---|
| CVIAN spatial-v1 label-cost selector | Complete; test consumed | Development NO-GO; no confirmatory claim |
| CVIAN four-role label-cost selector | Complete; hash-chain verified; test consumed | Development/exploratory within-CVIAN NO-GO |
| Milton frozen zero-shot transfer | Complete; independently verified | Sensitivity only; no GO/NO-GO decision and no external confirmation |
| RQ1 Study A, CVIAN anatomy | Complete on development roles | 0/3 pre-specified criteria met |
| RQ1 Study B, Eaton damaged-component observability v1 | Frozen development execution | Outcome-blind structural STOP (28 total blocks < 30 eligible-block minimum); no H-B1/H-B2 or confirmation result |
| RQ2 view × field attestability matrix | Dependency audit complete | Strict current-data NO-GO; required labels/annotations absent |
| RQ3 attestation-coverage acquisition | Dependency audit complete | Strict current-data NO-GO; calibrated RQ2 matrix and held-out labels absent |
| RQ4 calibrated stop/defer | Prerequisites audited | BLOCKED; current rule is risk-aware heuristic only |

## 1. CVIAN four-role label-cost result

The repaired four-role protocol improves role accounting but does not create an
external or never-seen confirmation. Its final role is
`selector_selection_holdout_with_historical_base_exposure`: 137 samples in 22
sequence-connected components, including 21 severe samples. The frozen base had
historical exposure, so the claim scope remains within-CVIAN development.

At the registered fixed budget `k=3`, contrast is defined as baseline cost minus
utility-policy cost; positive values would favor the utility policy.

| Comparator | Component-macro contrast | 95% CI | Five-seed lower-cost requirement | Severe-miss non-inferiority |
|---|---:|---:|---|---|
| farthest angular coverage | `-0.050935` | `[-0.114026, -0.000229]` | not met; utility lower in 1/5 seeds | not met |
| privileged maximum building | `-0.025325` | `[-0.054784, 0.001634]` | not met; utility lower in 2/5 seeds | not met |

The primary decision is therefore **NO-GO**. Adaptive stop/defer is secondary:
there is no formal stop calibration, fail-closed STOP is disabled, and the
threshold-1 rule is only a risk-aware heuristic. See the hash-bound
[`CVIAN result`](./cvian_sequence_utility_v1.md).

### CRLF completion-verifier downgrade

The completed-run verifier compared the stored commitment file against a
canonical LF rendering, while the file on disk had 105 CRLF newlines:

- stored raw commitment SHA-256:
  `edb9044d231866ccd5f5b06a4d1bc0bd6f44d6d5ef925f131bfa7ca997d37009`;
- canonical LF rendering SHA-256 expected by that verifier:
  `c3011406c3ff6c5fb3ed1da34dec73489f1101987d90ab018e2b60548a1c9b22`;
- unchanged semantic fingerprint:
  `0d5bddc735dd819dc379c212f7082745aeeceeadc2667c8fd5144f57fe7f9b42`.

The defect is confined to newline representation in the completed-run verifier;
it did not affect original scoring. Every downstream raw-hash DAG edge uses the
actual CRLF SHA. An independent verifier returned `verified=true` with
`verification_defect="windows_newline_hash_mismatch"` after independently
recomputing the raw hash chain, primary NO-GO, both 10,000-draw bootstraps, and
the report. It did no inference, prospective NPZ semantic loading, or rescoring.

The only correct final wording is:

> **hash-chain-verified consumed development/exploratory within-CVIAN NO-GO**

This is not a confirmatory or external-event result. See the
[`integrity addendum`](./cvian_sequence_utility_v1_integrity_addendum.md) and the
machine-readable
[`completion verification`](../../outputs/cvian_sequence_active_v2/utility_experiment/completion_verification.json).

## 2. Milton one-time sensitivity

Milton was used once with a frozen zero-shot, forward-only scorer. There was no
Milton fitting, calibration, threshold selection, or policy selection. Milton
influenced hypothesis development and lacks an observed acquisition sequence and
trusted compass, so it is not an external confirmation event. In addition, P0.3
remains unresolved: the manifest/source-split and hash-chain audits establish
lineage inside this local scoring run, but do not establish whether post-SVI
media were collected or generated. Retained `prompt` metadata does not resolve
that collected-vs-generated provenance ambiguity.

All contrasts below are comparator minus relative-policy component-macro cost;
positive values favor the relative policy. The primary interval clusters 11
joint dependency-group/spatial-block components.

| Comparator | Dependency-group contrast (95% CI) | Spatial-block contrast (95% CI) | Interpretation |
|---|---:|---:|---|
| `random_mc32` | `+0.024557` (`[0.013279, 0.044898]`) | `+0.021502` (`[0.013730, 0.034251]`) | positive descriptive signal |
| farthest | `+0.007531` (`[-0.012983, 0.040839]`) | `+0.006218` (`[-0.009734, 0.029054]`) | interval crosses zero |
| clockwise | `+0.061382` (`[-0.060293, 0.159261]`) | `+0.068581` (`[-0.040095, 0.147292]`) | interval crosses zero |

The random worst-rotation contrast is negative, so the random comparison is not
uniform across rotations. Absolute-aware is an off-support OOD diagnostic;
max-confidence and label-oracle policies are privileged and non-headline.

The completed sensitivity contains 1,707 samples, five seeds, eight origins, 259
dependency groups, and 57 spatial blocks. The independent completion verifier
returned `verified=true` with no defect or downgrade. It independently matched
the full hash chain, 2,594,640 decisions, 2,184,960 RNG replays, stage estimands,
and the 10,000-draw joint and secondary bootstraps without inference, semantic
NPZ loading, or rescoring. The final status is
`sensitivity_complete_no_confirmation`: the positive random contrast is useful
descriptive evidence, but there is **no GO and no confirmatory transfer claim**.
See the hash-bound [`Milton result`](./active_view_milton_zero_shot_v1.md) and
human-readable
[`integrity verification`](./active_view_milton_zero_shot_v1_integrity.md). The
ignored machine output is retained as the
[`independent verification JSON`](../../outputs/cvian_sequence_active_v2/milton_zero_shot_sensitivity/independent_completion_verification.json).

## 3. Attestability branch

### RQ1 Study A: 0/3

The CVIAN development study used `gate_fit` and `risk_calibration` only; it did
not touch `final_test`. None of the three pre-specified criteria was met:

- H-A1 directional asymmetry: not met;
- H-A2 visibility-conditioned correctness: not met;
- H-A3 visibility predicts held-out correctness: not met.

The existing four visibility features do not carry held-out attestability
signal. A weak street-more-severe direction appears in 5/5 seeds but remains
unconfirmed. With only seven spatial blocks, the study is structurally
underpowered for in-event confirmation. This is not evidence that disagreement
is information; it shows that the current feature set is insufficient. See
[`CVIAN disagreement anatomy`](./disagreement_anatomy_cvian_v1.md).

### RQ1 Study B: Eaton blocked at the construct boundary

The inventory statements in this historical subsection describe the state
before the focused v1 protocol. Predictions and frozen spatial roles now exist;
the current registered endpoint is the structural development STOP linked
above. The missing genuine human damage-dominance construct remains relevant,
but v1 does not request it because the block-capacity gate already cannot pass.

The Eaton audit joined 19,776 of 19,780 manifest rows to DINS records, covering
18,411 unique structures. It cannot run the registered component-anatomy test:

- there is no component-damage presence/dominance reference;
- DINS component-like fields represent construction, material, presence, or
  exposure, not damage dominance;
- there is no genuine Eaton street/remote severity-prediction export; and
- there is no frozen Eaton spatial split/group artifact.

Fire origin cannot substitute for damage dominance. Study B is therefore
**inferentially BLOCKED, not empirically negative**. See the
[`Eaton feasibility audit`](./disagreement_anatomy_eaton_v1.md).

### RQ2/RQ3: upstream dependency NO-GO

CVIAN does not contain real per-field damage references, per-view
observability/judgment/abstention annotations, field-level predictions,
geometry/occlusion, frozen field weights, or an attestation-labeled held-out
test. Eaton likewise lacks component-damage labels, view annotations,
predictions, frozen roles/groups, geometry, an action protocol, and an
attestation-labeled test.

Consequently:

- RQ2 empirical view × field matrix training is a strict current-data NO-GO;
- RQ3 attestation-coverage utility training/evaluation is a strict current-data
  NO-GO; and
- the existing label-cost utility must not be renamed attestation coverage.

No fake labels or synthetic construct substitutes were created. These are
upstream dependency decisions, not negative empirical tests of the central
hypothesis. See the
[`attestation dependency gate`](./attestation_dependency_gate_20260710.md),
[`annotation contract`](../../configs/crossviewguard_attestation_contract_v1.json),
and
[`dependency inventory`](../../configs/crossviewguard_attestation_dependency_inventory_20260710.json).

## 4. What is finished and what is not

Finished on current assets:

- CVIAN georeference/split repair and label-cost active-view evaluation;
- the repaired four-role CVIAN NO-GO and its integrity accounting;
- the one-time frozen Milton sensitivity and independent completion verification;
- RQ1 CVIAN development anatomy and Eaton construct/data feasibility audit; and
- executable RQ2/RQ3 contracts, inventories, and fail-closed dependency gates.

Not finished—and not executable without new evidence:

- confirmation that disagreement encodes view-conditional attestability;
- a calibrated field-level view × field attestability matrix;
- a valid attestation-coverage acquisition policy comparison;
- an independently calibrated, risk-controlled attestation stop/defer policy;
  and
- external/prospective validation on real acquisition sequences.

Thus the central hypothesis is **unresolved**, not validated and not falsified.

## 5. External-data unlock conditions

### Eaton / RQ1

Provide all of the following before H-B1/H-B2:

- genuine predictions unique on `(seed,pair_id)`;
- a frozen spatial-block artifact; and
- provenance-bearing `component_dominance ∈ {roof,facade,mixed,unknown}` with
  `reference_semantics=damage_dominance`.

### RQ2

Before model fitting:

- a real provenance-bearing annotation artifact must pass the frozen contract;
- a field × view × role support/power audit must pass;
- the dependence-aware fit/calibration plan must be frozen; and
- geometry claims require camera-building distance/bearing, orientation, and
  occlusion.

### RQ3

Before policy evaluation:

- use a frozen, calibrated, out-of-role RQ2 matrix;
- freeze decision-field weights and costs before test;
- isolate policy-fit, validation, and test dependencies; and
- reserve a prospective/external attestation-labeled test. Any defect consumes
  that test.

### RQ4 and external acquisition

Risk-controlled STOP/defer additionally requires enough independent calibration
groups and severe cases, a pre-registered loss/`alpha`/`delta`/threshold grid,
and target-event calibration or a justified shift-aware protocol. A real
next-view acquisition claim requires new multi-azimuth field/vehicle/UAV data or
an equivalent prospective sequence, not another reveal order over the same
consumed panorama test. If Milton is ever proposed for more than sensitivity,
its post-SVI collected-vs-generated media provenance must first be resolved by a
source-authoritative artifact; manifest and hash-chain integrity alone are not
sufficient.

## 6. Consumed-data and claim ledger

- Do not tune selector architecture, budget, threshold, comparator set, or GO
  rule on the consumed CVIAN spatial-v1/four-role tests.
- Do not repeat Milton v1 scoring to select a policy; it is a completed one-time
  sensitivity.
- Do not describe Milton P0.3 media provenance as resolved merely because the
  local manifest/hash chain verifies.
- Do not spend CVIAN RQ1 `final_test` merely to overcome the seven-block power
  limitation.
- Do not treat Eaton's missing construct as a zero effect.
- Do not relabel label-cost utility as attestation coverage.
- Reopen the central branch only after the unlock artifacts exist and a new
  protocol is frozen before outcome inspection.

The active-evidence plan and attestability revision should therefore be read as
a protocol history plus this execution boundary, not as evidence that the full
CrossViewGuard idea has already been executed:

- [`active-evidence research plan`](../2026-07-10_crossviewguard_active_evidence_research_plan.md)
- [`attestability revision`](../2026-07-10_crossviewguard_attestability_revision.md)
