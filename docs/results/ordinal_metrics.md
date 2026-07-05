# Ordinal Metrics (QWK / MAE)

The 3-class damage tasks are ordinal: confusing `no_or_trace_damage` with
`destroyed` is operationally worse than confusing adjacent classes.
Quadratic weighted kappa (QWK) and MAE reflect this; `extreme_error_rate`
is the fraction of two-step errors (class 0 predicted as 2 or vice versa).
Mean +/- std over seeds, main 3-class protocol.

| dataset | method | QWK | MAE | extreme_error_rate |
| --- | --- | --- | --- | --- |
| altadena_3class | concat | 0.9500 +/- 0.0099 | 0.0880 +/- 0.0185 | 0.0034 +/- 0.0006 |
| altadena_3class | crossview | 0.9549 +/- 0.0055 | 0.0780 +/- 0.0116 | 0.0040 +/- 0.0018 |
| altadena_3class | late_fusion_probability_average | 0.9484 +/- 0.0060 | 0.0860 +/- 0.0127 | 0.0062 +/- 0.0015 |
| altadena_3class | remote_only | 0.9253 +/- 0.0039 | 0.1200 +/- 0.0098 | 0.0104 +/- 0.0030 |
| altadena_3class | street_only | 0.9278 +/- 0.0097 | 0.1136 +/- 0.0169 | 0.0111 +/- 0.0035 |
| ian_original | concat | 0.7111 +/- 0.0100 | 0.3400 +/- 0.0088 | 0.0278 +/- 0.0038 |
| ian_original | crossview | 0.7340 +/- 0.0427 | 0.2989 +/- 0.0402 | 0.0200 +/- 0.0067 |
| ian_original | late_fusion_probability_average | 0.7301 +/- 0.0108 | 0.2922 +/- 0.0139 | 0.0211 +/- 0.0051 |
| ian_original | remote_only | 0.6336 +/- 0.0375 | 0.4000 +/- 0.0384 | 0.0456 +/- 0.0084 |
| ian_original | street_only | 0.7121 +/- 0.0099 | 0.3078 +/- 0.0107 | 0.0167 +/- 0.0058 |
| milton_original | concat | 0.7719 +/- 0.0313 | 0.2507 +/- 0.0402 | 0.0013 +/- 0.0023 |
| milton_original | crossview | 0.7678 +/- 0.0169 | 0.2559 +/- 0.0136 | 0.0013 +/- 0.0023 |
| milton_original | late_fusion_probability_average | 0.7608 +/- 0.0062 | 0.2730 +/- 0.0060 | 0.0013 +/- 0.0023 |
| milton_original | remote_only | 0.7015 +/- 0.0227 | 0.3281 +/- 0.0227 | 0.0105 +/- 0.0082 |
| milton_original | street_only | 0.7668 +/- 0.0147 | 0.2598 +/- 0.0205 | 0.0000 +/- 0.0000 |
