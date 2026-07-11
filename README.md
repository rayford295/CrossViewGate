# CrossViewGate

**Trust the view that sees the target: visibility-conditioned reliability
gating for conflict-aware cross-view disaster damage assessment.**

Project website: https://rayford295.github.io/CrossViewGate/

Cross-view fusion of ground-level and overhead imagery is standard practice in
post-disaster building damage assessment — and almost always *symmetric*: both
views are trusted equally everywhere. This repository shows that assumption
discards most of the useful information on the cases that matter, and provides
the method, the causal evidence, and the operational payoff for doing better.

Manuscript (ISPRS J. Photogramm. Remote Sens. target):
`paper/isprs_manuscript.md`.

## Findings

1. **The oracle gap.** On *conflict cases* — samples where independently
   trained street-view and overhead models disagree — an oracle that picks the
   correct single view beats every tested fusion method by **0.37–0.41
   accuracy**, consistently across three disasters (2025 Eaton wildfire,
   Hurricane Ian, Hurricane Milton), five seeds, calibration, and backbones.
   Symmetric fusion leaves most view-reliability information unused.

2. **The reliability gate.** A **linear** gate over 11 interpretable features
   — SegFormer building-visibility, calibrated per-view confidence/entropy,
   and cross-view disagreement — mixes street/remote/crossview probabilities
   per sample. It is the only method that significantly beats calibrated
   probability averaging (wildfire conflicts +0.051, p = 0.0001) and beats
   end-to-end fusion on conflicts (+0.072, p < 1e-4) and the full test set
   (+0.018, p < 1e-4), with parity (never significant degradation) on the
   panoramic datasets. The learned rule reads: *trust the street view when it
   is confident and the building is centered in the frame.*

3. **Causal mechanism.** Cropping hurricane panoramas to building-centered
   90° views doubles the fusion benefit on conflicts (Milton oracle-gap
   closure 0.177 → 0.365); geometrically identical random crops do not
   (6/6 dataset-seed pairs). Street-view *target alignment* — not disaster
   type — controls the value of cross-view fusion.

4. **Label-free damage mapping.** The spatial density of cross-view conflicts
   predicts tile-level damage without annotations (wildfire Spearman
   r = 0.615, p = 0.001) while single-view uncertainty density does not
   (it *anti-correlates* on wildfire).

5. **A negative result worth knowing.** Once single-view models are converged
   and temperature-calibrated, simple probability averaging matches or beats
   end-to-end learned fusion. The advantage of learning lies in asymmetric,
   reliability-aware arbitration — not in fusion per se.

## Research Roadmap

The next research stage moves beyond damage classification toward
**risk-controlled active evidence acquisition**: decide when the available
evidence is sufficient, which view to trust, which view to acquire next, when
to defer to a human, and where limited field-inspection resources should go.

See the dated research plan:
[`CrossViewGuard: Risk-Controlled Active Evidence Acquisition`](docs/2026-07-10_crossviewguard_active_evidence_research_plan.md).

The first CVIAN spatial-v1 active-view development benchmark is now complete.
Its privileged greedy one-step, label-aware reference shows a large gap, but it
is not a globally optimal acquisition upper bound. The current supervised
selector does not outperform simple coverage or privileged building heuristics
at the three-view budget, so the result is reported as a transparent No-Go
rather than a positive active-policy claim. The model/selector split isolates
only the downstream heads because the frozen encoder saw the complete training
role. The spatial development test is now consumed; future selector variants
require out-of-fold/base-encoder role isolation and a new sequence- or
event-held-out confirmatory test. See
[`active_view_cvian_spatial_v1.md`](docs/results/active_view_cvian_spatial_v1.md).

## Repository map

```text
CrossViewGate/
|- crossview_conflict/        # package: data, models, training, utils
|- scripts/                   # pipeline + analysis (see below)
|- paper/                     # manuscript, figures, references
|- docs/
|  |- results/                # result documents (v2 = headline protocol)
|  |- archive/                # development-era planning/analysis notes
|  `- datasets.md
|- data/                      # local manifests/splits (images not tracked)
`- outputs/                   # run outputs (not tracked)
```

Key scripts, in pipeline order:

| Stage | Script |
| --- | --- |
| Build splits (grouped, anti-leakage) | `make_group_splits.py`, `build_*_manifests.py` |
| Train / evaluate base models | `train_triage.py`, `eval_triage.py`, `run_multiseed_v2.sh` |
| Visibility features | `extract_visibility_features.py` |
| Calibration decomposition | `analyze_calibration_fusion.py` |
| Reliability gate (+ transfer) | `train_reliability_gate.py` |
| Causal FOV intervention | `build_fov_intervention.py`, `run_fov_intervention.sh`, `analyze_fov_intervention.py` |
| Conflict-density maps | `analyze_conflict_density_maps.py` |
| Development active next-view benchmark | `build_cvian_active_view_manifests.py`, `cache_cvian_active_view_embeddings.py`, `run_cvian_active_view_experiment.py`, `summarize_cvian_active_view.py` |
| Statistics | `analyze_pooled_seed_tests.py`, `analyze_ordinal_metrics.py` |
| Manuscript | `build_manuscript_docx.py` |

## Reproducing

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 1. converged 5-seed suite (trains 60 models, writes test+val predictions)
bash scripts/run_multiseed_v2.sh

# 2. visibility features (one SegFormer-B0 pass per street image)
python scripts/extract_visibility_features.py --split-csv <split>/val.csv --output-csv outputs/analysis/visibility_features/<ds>_val.csv

# 3. analyses (all read predictions; no retraining)
python scripts/analyze_calibration_fusion.py  --multiseed-root outputs/multiseed_v2 --seeds 42,123,456,789,1011
python scripts/train_reliability_gate.py      --multiseed-root outputs/multiseed_v2 --seeds 42,123,456,789,1011
python scripts/analyze_pooled_seed_tests.py   --multiseed-root outputs/multiseed_v2 --seeds 42,123,456,789,1011
```

Before rebuilding the Ian split, place the official CVIAN position file at
`<IAN_hurricane>/02_Position/CVIAN_position.geojson` and the release
`checksums.sha512` at `<IAN_hurricane>/checksums.sha512`, then run:

```powershell
python scripts/georeference_ian_hurricane.py --dataset-root <IAN_hurricane>
python scripts/build_ian_hurricane_manifests.py `
  --dataset-root <IAN_hurricane> `
  --output-dir data/splits/ian_hurricane_original `
  --split-strategy spatial-block --spatial-buffer-m 25
```

The georeference and repaired-split audit is documented in
[`docs/results/cvian_georeference_split_audit.md`](docs/results/cvian_georeference_split_audit.md).

The repaired Ian pipeline continues with a fingerprinted five-seed run and a
strict gate-fit/risk-calibration/final-test split:

```powershell
python scripts/run_ian_spatial_multiseed.py --jobs 2
python scripts/build_risk_protocol_manifests.py `
  --split-dir data/splits/ian_hurricane_original `
  --output-dir data/splits/ian_hurricane_risk_protocol `
  --event-id hurricane_ian_cvian
```

Selective-triage claims require explicit spatial/group auditing and one fixed
gate identity. The executed result and fail-closed routing case are summarized
in [`docs/results/selective_triage_ian_spatial_v1.md`](docs/results/selective_triage_ian_spatial_v1.md).

Datasets are kept local (see `docs/datasets.md`): Eaton/Altadena wildfire
inspection pairs (CAL FIRE DINS), the CVIAN Hurricane Ian release
(Li et al., 2025), and the Milton street-view/satellite pairing. Eaton splits
are grouped by object id because random splits leak. The current CVIAN/Milton
split provenance and spatial-grouping audit is documented in the research
roadmap above.

## Results documents

See `docs/results/README.md` for the full index. Headline numbers come from
the converged 5-seed `_v2` documents; 3-epoch pilot results are retained as a
low-budget ablation.

## Naming note

This repository was previously named `CrossViewConflict`; old URLs redirect.
The Python package keeps the import name `crossview_conflict`.

## License

See `LICENSE`.
