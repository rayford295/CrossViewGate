# Ordinal Metrics (QWK / MAE)

The 3-class damage tasks are ordinal: confusing `no_or_trace_damage` with
`destroyed` is operationally worse than confusing adjacent classes.
Quadratic weighted kappa (QWK) and MAE reflect this; `extreme_error_rate`
is the fraction of two-step errors (class 0 predicted as 2 or vice versa).
Mean +/- std over seeds, main 3-class protocol.

| dataset | method | QWK | MAE | extreme_error_rate |
| --- | --- | --- | --- | --- |
| altadena_3class | concat | 0.9569 +/- 0.0064 | 0.0741 +/- 0.0109 | 0.0043 +/- 0.0008 |
| altadena_3class | crossview | 0.9628 +/- 0.0046 | 0.0637 +/- 0.0100 | 0.0040 +/- 0.0015 |
| altadena_3class | late_fusion_probability_average | 0.9668 +/- 0.0102 | 0.0545 +/- 0.0187 | 0.0049 +/- 0.0013 |
| altadena_3class | remote_only | 0.9471 +/- 0.0135 | 0.0810 +/- 0.0243 | 0.0103 +/- 0.0008 |
| altadena_3class | street_only | 0.9495 +/- 0.0070 | 0.0780 +/- 0.0145 | 0.0094 +/- 0.0035 |
| ian_original | concat | 0.7237 +/- 0.0260 | 0.3167 +/- 0.0288 | 0.0260 +/- 0.0043 |
| ian_original | crossview | 0.7545 +/- 0.0122 | 0.2793 +/- 0.0148 | 0.0140 +/- 0.0043 |
| ian_original | late_fusion_probability_average | 0.7451 +/- 0.0088 | 0.2920 +/- 0.0073 | 0.0167 +/- 0.0067 |
| ian_original | remote_only | 0.6661 +/- 0.0148 | 0.3793 +/- 0.0198 | 0.0273 +/- 0.0064 |
| ian_original | street_only | 0.7149 +/- 0.0152 | 0.3187 +/- 0.0243 | 0.0213 +/- 0.0051 |
| milton_original | concat | 0.7820 +/- 0.0115 | 0.2346 +/- 0.0129 | 0.0008 +/- 0.0018 |
| milton_original | crossview | 0.7892 +/- 0.0130 | 0.2236 +/- 0.0153 | 0.0000 +/- 0.0000 |
| milton_original | late_fusion_probability_average | 0.7958 +/- 0.0076 | 0.2268 +/- 0.0126 | 0.0000 +/- 0.0000 |
| milton_original | remote_only | 0.7280 +/- 0.0136 | 0.2906 +/- 0.0151 | 0.0110 +/- 0.0051 |
| milton_original | street_only | 0.7824 +/- 0.0093 | 0.2409 +/- 0.0170 | 0.0000 +/- 0.0000 |
