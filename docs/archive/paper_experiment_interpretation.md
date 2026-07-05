# Paper Experiment Interpretation

The strongest framing is not "cross-view always beats single-view." The stronger
and more defensible claim is:

Cross-view models improve disaster damage assessment specifically when the two
views disagree and the street-view evidence is target-aligned. The benefit is
largest for property-level wildfire imagery, weaker and more split-sensitive for
IAN, and moderate for Milton. Simple fusion baselines can be competitive, so the
paper should explain when learned cross-view interaction helps rather than only
reporting that it helps.

## Evidence Completed

Main protocol: wildfire 3-class plus IAN and Milton 3-class, each with
`street_only`, `remote_only`, `concat`, and `crossview` over seeds 42, 123, and
456. Results are in `docs/multiseed_main_results.md`.

Conflict statistics: bootstrap confidence intervals, paired permutation tests,
and McNemar tests compare `crossview - best_single_view` on conflict subsets.
Altadena is positive and mostly significant; IAN is positive but seed-sensitive;
Milton is positive but underpowered on the current conflict subset.

Label robustness: the wildfire 6-class audit keeps the original labels and still
shows crossview beating the best single view on conflicts in 3/3 seeds. The old
binary sensitivity setting is easier and has a much smaller conflict subset:
crossview strongly beats remote-only, but does not significantly beat the best
single view or concat. This is useful because it shows the 3-class result is not
just an artifact of label merging.

Mechanism validation: building visibility proxies explain the dataset
heterogeneity. Altadena has high building pixel ratio and target visibility, IAN
has very low street-view building visibility, and Milton is intermediate.

Stronger baselines: late probability averaging, logit averaging, confidence
voting, concat, and ResNet50 sanity checks are included. The results show that
late fusion can be strong in calibration-friendly settings, while learned
cross-view training is more useful on hard conflict cases and when alignment is
better.

## More Incisive Paper Claims

1. Conflict resolution is the right evaluation target. Overall accuracy hides the
   central problem because most samples are easy or single-view-consistent.

2. The remaining gap to the oracle single-view upper bound is large. This means
   there is unresolved view reliability information: a model that learns when to
   trust each view could improve substantially.

3. Label granularity changes what "fusion" means. Six-class wildfire labels
   expose sparse-class noise and calibration effects, while binary labels make
   the task too easy and shrink the conflict subset.

4. Cross-view gains are alignment-conditioned. The mechanism result lets the
   paper explain why Altadena, Milton, and IAN differ instead of treating
   heterogeneity as an inconvenience.

5. Backbone sanity argues against an overbroad claim. ResNet50 improves
   crossview on IAN and Milton conflict subsets but not Altadena, where
   street-only wins. This suggests the contribution should focus on protocol,
   conflict-aware evaluation, and alignment mechanisms, not a universal
   architecture claim.

## Best Next Experiments

The next most valuable experiment is an alignment-conditioned gate: predict
view reliability from building visibility, street-view target alignment, and
single-view confidence, then compare it against crossview and late fusion on
conflict subsets.

A second strong experiment is calibration-aware fusion: temperature-scale the
single-view models on the validation set, then rerun probability/logit averaging.
This will clarify whether IAN's late-fusion strength is mostly calibration.

A third is an ordinal/cost-sensitive loss for 3-class severity. Damage classes
are ordered, so confusing `no_or_trace_damage` with `destroyed` should cost more
than confusing adjacent classes. This is especially relevant for wildfire 6-class
and 3-class comparisons.

For a top-tier submission, the story should be: conflict-aware cross-view
assessment works, but only a mechanism-aware evaluation reveals when and why it
works.
