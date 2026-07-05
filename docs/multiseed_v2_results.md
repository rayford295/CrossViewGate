# Multiseed v2: Converged 5-Seed Suite (Headline Protocol)

60 runs: 3 datasets x 4 modes x seeds {42, 123, 456, 789, 1011}, epochs 15
with patience-4 early stopping, otherwise matched to the v1 protocol
(`scripts/run_multiseed_v2.sh`). All v1 analyses were rerun against this
suite; per-analysis docs: `calibration_decomposition_v2.md`,
`reliability_gate_results_v2.md`, `pooled_seed_tests_v2.md`,
`ordinal_metrics_v2.md`.

## Main table (test, mean +/- std over 5 seeds)

| dataset | method | accuracy | macro_f1 | conflict_acc |
| --- | --- | --- | --- | --- |
| altadena_3class | street_only | 0.9314 +/- 0.0165 | 0.7219 | 0.4863 |
| altadena_3class | remote_only | 0.9292 +/- 0.0243 | 0.6996 | 0.4747 |
| altadena_3class | concat | 0.9302 +/- 0.0105 | 0.7062 | 0.6739 |
| altadena_3class | crossview | 0.9403 +/- 0.0110 | 0.7162 | 0.6988 |
| altadena_3class | calibrated prob average | 0.9537 +/- 0.0138 | 0.7220 | 0.7317 |
| altadena_3class | **gate3_linear** | **0.9588** | — | **0.7678** |
| ian_original | street_only | 0.7027 +/- 0.0202 | 0.7060 | 0.5290 |
| ian_original | remote_only | 0.6480 +/- 0.0146 | 0.6497 | 0.3667 |
| ian_original | concat | 0.7093 +/- 0.0283 | 0.7092 | 0.5879 |
| ian_original | crossview | 0.7347 +/- 0.0145 | 0.7391 | 0.6240 |
| ian_original | calibrated prob average | 0.7313 +/- 0.0051 | 0.7346 | 0.6166 |
| ian_original | gate3_linear | 0.7307 | — | 0.6180 |
| milton_original | street_only | 0.7591 +/- 0.0170 | 0.7650 | 0.5566 |
| milton_original | remote_only | 0.7205 +/- 0.0124 | 0.7259 | 0.4072 |
| milton_original | concat | 0.7661 +/- 0.0123 | 0.7711 | 0.6218 |
| milton_original | crossview | 0.7764 +/- 0.0153 | 0.7808 | 0.6493 |
| milton_original | calibrated prob average | 0.7764 +/- 0.0146 | 0.7821 | 0.6257 |
| milton_original | gate3_linear | 0.7772 | — | 0.6309 |

Oracle-gap closure on conflicts: gate3 0.524 / crossview 0.346 (Altadena);
0.234 / 0.252 (IAN); 0.181 / 0.222 (Milton).

## What the pooled tests establish (v2, sign-flip permutation)

1. **Cross-view fusion beats both single views on conflicts, significant on
   all three datasets** (worst case p = 0.013). Also significant on the full
   test set for Altadena and IAN.
2. **gate3_linear significantly beats crossview on Altadena**: conflicts
   +0.072 (p < 1e-4), full test +0.018 (p < 1e-4); parity (ns) on IAN and
   Milton, never significantly worse.
3. **Head-to-head vs the strongest baseline** (calibrated probability
   averaging, computed post-hoc): gate3 +0.051 on Altadena conflicts
   (95% CI [+0.026, +0.077], p = 0.0001); parity on IAN/Milton.

## Changes vs the 3-epoch v1 suite (honest accounting)

- **Strengthened**: gate3's wildfire advantage roughly doubles under
  convergence (+0.043 -> +0.072 on conflicts); wildfire conflict-density
  correlation rises (hard r = 0.545 -> 0.615, p = 0.001); crossview's
  conflict advantage over single views becomes significant everywhere.
- **Weakened / reframed**: vanilla crossview no longer beats calibrated
  probability averaging under convergence (Altadena conflict means 0.699 vs
  0.732). The claim "learned fusion beats simple fusion" belongs to the
  **gate**, not to vanilla end-to-end fusion. The paper should say exactly
  that: with converged single-view models, calibrated averaging is a strong
  baseline that vanilla learned fusion does not reliably beat, and the
  reliability gate is what beats it, precisely where street views are
  target-aligned.
- **Weakened**: Milton conflict-density correlation drops under convergence
  (soft r = 0.485 -> 0.289, ns). Converged single-view models disagree less,
  thinning the density signal. Report the hurricane density map as
  regime-sensitive supporting evidence, wildfire as the robust case.

## Protocol note

v1 (3-epoch) results remain in the repo as a low-budget ablation; all paper
headline numbers should come from this v2 suite.
