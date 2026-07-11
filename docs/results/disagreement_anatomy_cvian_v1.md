# RQ1 Study A results: CVIAN disagreement anatomy (development, v1)

## Claim scope

Exploratory/development analysis executed per
[`disagreement_anatomy_protocol.md`](disagreement_anatomy_protocol.md) on the
p0.2-v1 evidence exports, roles `gate_fit` + `risk_calibration` only
(`final_test` untouched). Five seeds, 2,000-replicate cluster bootstrap over
`spatial_block_id`. **None of the three pre-specified development criteria was
met.** Artifacts and input SHA-256 fingerprints:
`outputs/analysis/disagreement_anatomy_ian_spatial_v1/` (not tracked; regenerate
with the command in the protocol).

## Headline numbers

Disagreement is the norm, not the exception: 40–50% of role rows disagree
across seeds, and ~85% of disagreements are decidable (exactly one view
correct). Street is the correct view in 54–62% of decidable cases.

| Seed | Disagree rate | Decidable n | H-A1 share (CI) | H-A2 med. diff (CI) | H-A3 vis-only AUROC | H-A3 conf-only AUROC |
| ---: | ---: | ---: | --- | --- | ---: | ---: |
| 42 | 0.498 | 178 | 0.441 (0.216, 0.611) | 0.008 (−0.002, 0.016) | 0.540 | 0.507 |
| 123 | 0.395 | 153 | 0.309 (0.144, 0.579) | 0.002 (−0.011, 0.010) | 0.503 | 0.714 |
| 456 | 0.402 | 141 | 0.406 (0.247, 0.604) | 0.003 (−0.002, 0.011) | 0.492 | 0.716 |
| 789 | 0.463 | 169 | 0.463 (0.203, 0.610) | 0.003 (−0.006, 0.010) | 0.410 | 0.543 |
| 1011 | 0.463 | 164 | 0.337 (0.164, 0.465) | 0.008 (−0.004, 0.019) | 0.433 | 0.585 |

H-A1 share = fraction of disagreements where overhead reports *more* severe.

## Development verdicts (all not met)

- **H-A1 directional asymmetry: not met.** Point estimates lean the same way
  in 5/5 seeds — disagreements tilt toward *street reporting more severe*
  (share 0.31–0.46) — but the cluster CIs exclude 0.5 in only 1/5 seeds.
- **H-A2 visibility conditions correctness: not met.** The median
  `center_building_ratio` difference (street-correct minus remote-correct) is
  positive in 5/5 seeds, rank-biserial 0.07–0.18 all positive, but every CI
  includes 0.
- **H-A3 visibility explains held-out correctness: not met.** Visibility-only
  AUROC 0.41–0.54 — no transferable signal. Confidence-only reaches 0.51–0.72
  and combined never beats confidence-only.

## Honest interpretation

1. **The existing four visibility features do not carry the attestability
   signal.** Whatever the reliability gate exploits from visibility in-sample,
   it does not predict *which view is correct* out-of-role on disagreements.
   Correctness attribution is currently driven by calibrated confidence, which
   the gate already uses.
2. **A consistent weak directional lean exists but is unconfirmed.** All five
   seeds agree that street tends to report more severe than overhead on CVIAN
   disagreements — physically plausible for hurricanes (street sees facade and
   water-line damage; overhead sees surviving roofs) — but with only 7 spatial
   blocks (3 fit + 4 eval), the design cannot confirm it.
3. **Power is structural, not incidental.** The gate_fit/risk_calibration
   roles contain 7 spatial blocks total. Wide CIs are guaranteed at this
   cluster count; this mirrors the selective-triage certification failure and
   strengthens the case that in-event CVIAN cannot support confirmatory
   claims of this type.

## Implications for the attestability revision

- Study A neither confirms nor falsifies the attestability thesis: it shows
  the *current feature set* is insufficient, not that the structure is absent.
- **Study B (Eaton component anatomy) is now the critical test** — component
  fields are exactly the attestability signal that CVIAN's coarse
  segmentation-ratio features cannot express. Eaton also fixes the power
  problem (thousands of groups vs 7 blocks).
- Any future Study A iteration needs either better geometry features
  (per-sector occlusion, facade orientation relative to camera, distance) or
  should be folded into Study B.
- The directional-lean observation (street-more-severe on hurricane
  disagreements) is a concrete, cheap Study B hypothesis: it predicts the
  *opposite* lean on roof-dominated wildfire damage.

## Consumed-data ledger update

`gate_fit` and `risk_calibration` are now consumed for anatomy exploration as
pre-registered. `final_test` remains untouched.
