# Calibration Decomposition (Phase 3)

Per-view temperatures are fit on validation predictions and applied to test
logits. Fusion baselines are then recomputed on calibrated probabilities to
separate calibration effects from genuine complementary-information effects.
Conflict subsets use raw single-view argmax disagreement (temperature scaling
does not change single-view argmax, so the subset is identical before/after).
`oracle_gap_closure` is (method - best single view) / (oracle single view -
best single view) on the conflict subset.

## Per-view calibration quality

| dataset | mode | temperature | ECE raw | ECE calibrated |
| --- | --- | --- | --- | --- |
| altadena_3class | concat | 1.4255 +/- 0.3089 | 0.0232 +/- 0.0088 | 0.0126 +/- 0.0042 |
| altadena_3class | crossview | 1.4856 +/- 0.3660 | 0.0187 +/- 0.0098 | 0.0123 +/- 0.0031 |
| altadena_3class | remote_only | 1.4459 +/- 0.1531 | 0.0310 +/- 0.0148 | 0.0169 +/- 0.0041 |
| altadena_3class | street_only | 1.0948 +/- 0.3540 | 0.0291 +/- 0.0163 | 0.0140 +/- 0.0081 |
| ian_original | concat | 1.4008 +/- 0.5389 | 0.0928 +/- 0.0498 | 0.0532 +/- 0.0124 |
| ian_original | crossview | 1.5963 +/- 0.4212 | 0.1086 +/- 0.0370 | 0.0565 +/- 0.0076 |
| ian_original | remote_only | 1.4453 +/- 0.2506 | 0.0920 +/- 0.0433 | 0.0529 +/- 0.0153 |
| ian_original | street_only | 1.7498 +/- 0.3883 | 0.1399 +/- 0.0383 | 0.0625 +/- 0.0136 |
| milton_original | concat | 1.6671 +/- 0.6624 | 0.1086 +/- 0.0481 | 0.0540 +/- 0.0091 |
| milton_original | crossview | 1.7390 +/- 0.4550 | 0.1348 +/- 0.0441 | 0.0629 +/- 0.0138 |
| milton_original | remote_only | 1.0072 +/- 0.0475 | 0.0627 +/- 0.0218 | 0.0599 +/- 0.0138 |
| milton_original | street_only | 1.6947 +/- 0.5242 | 0.1020 +/- 0.0414 | 0.0593 +/- 0.0223 |

## Fusion methods, raw vs calibrated

| dataset | method | accuracy | macro_f1 | conflict_acc | oracle_gap_closure |
| --- | --- | --- | --- | --- | --- |
| altadena_3class | calibrated_confidence_voting | 0.9530 +/- 0.0129 | 0.7189 +/- 0.0141 | 0.7215 +/- 0.0247 | 0.4074 +/- 0.1236 |
| altadena_3class | calibrated_logit_average | 0.9538 +/- 0.0143 | 0.7258 +/- 0.0086 | 0.7341 +/- 0.0327 | 0.4372 +/- 0.1406 |
| altadena_3class | calibrated_probability_average | 0.9537 +/- 0.0138 | 0.7220 +/- 0.0100 | 0.7317 +/- 0.0281 | 0.4317 +/- 0.1314 |
| altadena_3class | concat_reference | 0.9302 +/- 0.0105 | 0.7062 +/- 0.0122 | 0.6739 +/- 0.0611 | 0.2929 +/- 0.1677 |
| altadena_3class | crossview_reference | 0.9403 +/- 0.0110 | 0.7162 +/- 0.0111 | 0.6988 +/- 0.0516 | 0.3462 +/- 0.1757 |
| altadena_3class | raw_confidence_voting | 0.9502 +/- 0.0192 | 0.7136 +/- 0.0160 | 0.7021 +/- 0.0672 | 0.3537 +/- 0.2349 |
| altadena_3class | raw_logit_average | 0.9510 +/- 0.0193 | 0.7201 +/- 0.0181 | 0.7115 +/- 0.0688 | 0.3770 +/- 0.2354 |
| altadena_3class | raw_probability_average | 0.9504 +/- 0.0188 | 0.7153 +/- 0.0175 | 0.7035 +/- 0.0615 | 0.3575 +/- 0.2210 |
| altadena_3class | remote_only | 0.9292 +/- 0.0243 | 0.6996 +/- 0.0191 | 0.4747 +/- 0.0966 | -0.2051 +/- 0.3115 |
| altadena_3class | street_only | 0.9314 +/- 0.0165 | 0.7219 +/- 0.0046 | 0.4863 +/- 0.0903 | -0.1650 +/- 0.2588 |
| ian_original | calibrated_confidence_voting | 0.7267 +/- 0.0071 | 0.7279 +/- 0.0073 | 0.6026 +/- 0.0083 | 0.1877 +/- 0.1271 |
| ian_original | calibrated_logit_average | 0.7327 +/- 0.0043 | 0.7366 +/- 0.0039 | 0.6207 +/- 0.0248 | 0.2359 +/- 0.1572 |
| ian_original | calibrated_probability_average | 0.7313 +/- 0.0051 | 0.7346 +/- 0.0058 | 0.6166 +/- 0.0132 | 0.2274 +/- 0.1105 |
| ian_original | concat_reference | 0.7093 +/- 0.0283 | 0.7092 +/- 0.0283 | 0.5879 +/- 0.0387 | 0.1494 +/- 0.1814 |
| ian_original | crossview_reference | 0.7347 +/- 0.0145 | 0.7391 +/- 0.0151 | 0.6240 +/- 0.0194 | 0.2517 +/- 0.0953 |
| ian_original | raw_confidence_voting | 0.7267 +/- 0.0062 | 0.7283 +/- 0.0062 | 0.6030 +/- 0.0209 | 0.1848 +/- 0.1652 |
| ian_original | raw_logit_average | 0.7300 +/- 0.0094 | 0.7339 +/- 0.0089 | 0.6129 +/- 0.0448 | 0.2115 +/- 0.2076 |
| ian_original | raw_probability_average | 0.7247 +/- 0.0073 | 0.7278 +/- 0.0064 | 0.5966 +/- 0.0302 | 0.1708 +/- 0.1614 |
| ian_original | remote_only | 0.6480 +/- 0.0146 | 0.6497 +/- 0.0174 | 0.3667 +/- 0.0493 | -0.4783 +/- 0.3379 |
| ian_original | street_only | 0.7027 +/- 0.0202 | 0.7060 +/- 0.0216 | 0.5290 +/- 0.0478 | 0.0000 +/- 0.0000 |
| milton_original | calibrated_confidence_voting | 0.7583 +/- 0.0190 | 0.7639 +/- 0.0179 | 0.5549 +/- 0.0607 | -0.0087 +/- 0.1228 |
| milton_original | calibrated_logit_average | 0.7819 +/- 0.0117 | 0.7876 +/- 0.0109 | 0.6468 +/- 0.0491 | 0.2169 +/- 0.1267 |
| milton_original | calibrated_probability_average | 0.7764 +/- 0.0146 | 0.7821 +/- 0.0134 | 0.6257 +/- 0.0489 | 0.1643 +/- 0.1206 |
| milton_original | concat_reference | 0.7661 +/- 0.0123 | 0.7711 +/- 0.0131 | 0.6218 +/- 0.0353 | 0.1567 +/- 0.0642 |
| milton_original | crossview_reference | 0.7764 +/- 0.0153 | 0.7808 +/- 0.0139 | 0.6493 +/- 0.0385 | 0.2216 +/- 0.0931 |
| milton_original | raw_confidence_voting | 0.7677 +/- 0.0139 | 0.7733 +/- 0.0136 | 0.5907 +/- 0.0186 | 0.0794 +/- 0.0653 |
| milton_original | raw_logit_average | 0.7764 +/- 0.0065 | 0.7819 +/- 0.0064 | 0.6242 +/- 0.0354 | 0.1624 +/- 0.1174 |
| milton_original | raw_probability_average | 0.7732 +/- 0.0126 | 0.7791 +/- 0.0119 | 0.6129 +/- 0.0471 | 0.1349 +/- 0.1070 |
| milton_original | remote_only | 0.7205 +/- 0.0124 | 0.7259 +/- 0.0117 | 0.4072 +/- 0.0366 | -0.3804 +/- 0.1999 |
| milton_original | street_only | 0.7591 +/- 0.0170 | 0.7650 +/- 0.0165 | 0.5566 +/- 0.0376 | 0.0000 +/- 0.0000 |

## Paired comparison vs crossview on the conflict subset

Per-seed paired bootstrap CIs and exact McNemar p-values against the
crossview reference; `sig` counts seeds with p < 0.05.

| dataset | method | delta vs crossview | McNemar sig |
| --- | --- | --- | --- |
| altadena_3class | calibrated_confidence_voting | 0.0228 +/- 0.0697 | 1/5 |
| altadena_3class | calibrated_logit_average | 0.0353 +/- 0.0754 | 2/5 |
| altadena_3class | calibrated_probability_average | 0.0329 +/- 0.0723 | 2/5 |
| altadena_3class | concat_reference | -0.0249 +/- 0.0789 | 2/5 |
| altadena_3class | raw_confidence_voting | 0.0033 +/- 0.0832 | 2/5 |
| altadena_3class | raw_logit_average | 0.0127 +/- 0.0855 | 2/5 |
| altadena_3class | raw_probability_average | 0.0048 +/- 0.0769 | 2/5 |
| altadena_3class | remote_only | -0.2240 +/- 0.0691 | 5/5 |
| altadena_3class | street_only | -0.2124 +/- 0.1322 | 3/5 |
| ian_original | calibrated_confidence_voting | -0.0214 +/- 0.0243 | 0/5 |
| ian_original | calibrated_logit_average | -0.0033 +/- 0.0365 | 0/5 |
| ian_original | calibrated_probability_average | -0.0074 +/- 0.0293 | 0/5 |
| ian_original | concat_reference | -0.0361 +/- 0.0314 | 0/5 |
| ian_original | raw_confidence_voting | -0.0210 +/- 0.0366 | 0/5 |
| ian_original | raw_logit_average | -0.0110 +/- 0.0565 | 0/5 |
| ian_original | raw_probability_average | -0.0274 +/- 0.0411 | 0/5 |
| ian_original | remote_only | -0.2573 +/- 0.0631 | 4/5 |
| ian_original | street_only | -0.0950 +/- 0.0416 | 1/5 |
| milton_original | calibrated_confidence_voting | -0.0945 +/- 0.0329 | 0/5 |
| milton_original | calibrated_logit_average | -0.0025 +/- 0.0403 | 0/5 |
| milton_original | calibrated_probability_average | -0.0236 +/- 0.0297 | 0/5 |
| milton_original | concat_reference | -0.0275 +/- 0.0205 | 0/5 |
| milton_original | raw_confidence_voting | -0.0587 +/- 0.0276 | 0/5 |
| milton_original | raw_logit_average | -0.0251 +/- 0.0520 | 0/5 |
| milton_original | raw_probability_average | -0.0364 +/- 0.0394 | 0/5 |
| milton_original | remote_only | -0.2421 +/- 0.0416 | 5/5 |
| milton_original | street_only | -0.0928 +/- 0.0471 | 0/5 |
