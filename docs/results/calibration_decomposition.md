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
| altadena_3class | concat | 1.1784 +/- 0.1248 | 0.0211 +/- 0.0095 | 0.0163 +/- 0.0039 |
| altadena_3class | crossview | 1.1206 +/- 0.0697 | 0.0148 +/- 0.0050 | 0.0188 +/- 0.0065 |
| altadena_3class | remote_only | 1.0781 +/- 0.1621 | 0.0332 +/- 0.0130 | 0.0251 +/- 0.0061 |
| altadena_3class | street_only | 0.8749 +/- 0.1555 | 0.0422 +/- 0.0220 | 0.0253 +/- 0.0134 |
| ian_original | concat | 1.4059 +/- 0.5101 | 0.0901 +/- 0.0272 | 0.0485 +/- 0.0152 |
| ian_original | crossview | 1.0580 +/- 0.0829 | 0.0514 +/- 0.0100 | 0.0440 +/- 0.0015 |
| ian_original | remote_only | 0.9859 +/- 0.0713 | 0.0666 +/- 0.0189 | 0.0618 +/- 0.0149 |
| ian_original | street_only | 1.3915 +/- 0.1785 | 0.1037 +/- 0.0083 | 0.0566 +/- 0.0198 |
| milton_original | concat | 1.0457 +/- 0.1224 | 0.0655 +/- 0.0066 | 0.0633 +/- 0.0155 |
| milton_original | crossview | 1.2041 +/- 0.2521 | 0.0715 +/- 0.0341 | 0.0433 +/- 0.0236 |
| milton_original | remote_only | 1.0035 +/- 0.0874 | 0.0890 +/- 0.0191 | 0.0897 +/- 0.0192 |
| milton_original | street_only | 0.9747 +/- 0.2647 | 0.0680 +/- 0.0044 | 0.0585 +/- 0.0135 |

## Fusion methods, raw vs calibrated

| dataset | method | accuracy | macro_f1 | conflict_acc | oracle_gap_closure |
| --- | --- | --- | --- | --- | --- |
| altadena_3class | calibrated_confidence_voting | 0.9202 +/- 0.0138 | 0.7023 +/- 0.0173 | 0.6701 +/- 0.0475 | 0.3131 +/- 0.1660 |
| altadena_3class | calibrated_logit_average | 0.9205 +/- 0.0133 | 0.7118 +/- 0.0237 | 0.6725 +/- 0.0496 | 0.3192 +/- 0.1742 |
| altadena_3class | calibrated_probability_average | 0.9212 +/- 0.0138 | 0.7080 +/- 0.0239 | 0.6774 +/- 0.0478 | 0.3307 +/- 0.1661 |
| altadena_3class | concat_reference | 0.9153 +/- 0.0189 | 0.7051 +/- 0.0097 | 0.6920 +/- 0.0813 | 0.3761 +/- 0.1839 |
| altadena_3class | crossview_reference | 0.9261 +/- 0.0128 | 0.7165 +/- 0.0037 | 0.6971 +/- 0.0895 | 0.3884 +/- 0.2185 |
| altadena_3class | raw_confidence_voting | 0.9209 +/- 0.0142 | 0.7044 +/- 0.0121 | 0.6751 +/- 0.0519 | 0.3248 +/- 0.1761 |
| altadena_3class | raw_logit_average | 0.9199 +/- 0.0127 | 0.7096 +/- 0.0143 | 0.6675 +/- 0.0491 | 0.3076 +/- 0.1752 |
| altadena_3class | raw_probability_average | 0.9202 +/- 0.0138 | 0.7053 +/- 0.0129 | 0.6702 +/- 0.0536 | 0.3133 +/- 0.1841 |
| altadena_3class | remote_only | 0.8905 +/- 0.0122 | 0.6729 +/- 0.0096 | 0.4532 +/- 0.0595 | -0.2036 +/- 0.2109 |
| altadena_3class | street_only | 0.8975 +/- 0.0178 | 0.7079 +/- 0.0166 | 0.5028 +/- 0.0798 | -0.0833 +/- 0.1443 |
| ian_original | calibrated_confidence_voting | 0.7267 +/- 0.0067 | 0.7304 +/- 0.0065 | 0.5970 +/- 0.0451 | 0.1285 +/- 0.1501 |
| ian_original | calibrated_logit_average | 0.7289 +/- 0.0107 | 0.7335 +/- 0.0099 | 0.6034 +/- 0.0301 | 0.1540 +/- 0.0469 |
| ian_original | calibrated_probability_average | 0.7244 +/- 0.0150 | 0.7287 +/- 0.0141 | 0.5900 +/- 0.0106 | 0.1105 +/- 0.1016 |
| ian_original | concat_reference | 0.6878 +/- 0.0069 | 0.6837 +/- 0.0081 | 0.5805 +/- 0.0407 | 0.0776 +/- 0.1847 |
| ian_original | crossview_reference | 0.7211 +/- 0.0379 | 0.7240 +/- 0.0371 | 0.6104 +/- 0.0854 | 0.1663 +/- 0.2322 |
| ian_original | raw_confidence_voting | 0.7233 +/- 0.0058 | 0.7276 +/- 0.0059 | 0.5874 +/- 0.0386 | 0.1035 +/- 0.1246 |
| ian_original | raw_logit_average | 0.7300 +/- 0.0186 | 0.7347 +/- 0.0177 | 0.6060 +/- 0.0154 | 0.1587 +/- 0.0759 |
| ian_original | raw_probability_average | 0.7289 +/- 0.0150 | 0.7332 +/- 0.0140 | 0.6030 +/- 0.0140 | 0.1489 +/- 0.0782 |
| ian_original | remote_only | 0.6456 +/- 0.0302 | 0.6437 +/- 0.0331 | 0.3608 +/- 0.0588 | -0.5579 +/- 0.4066 |
| ian_original | street_only | 0.7089 +/- 0.0150 | 0.7129 +/- 0.0138 | 0.5462 +/- 0.0494 | 0.0000 +/- 0.0000 |
| milton_original | calibrated_confidence_voting | 0.7126 +/- 0.0142 | 0.7206 +/- 0.0139 | 0.4863 +/- 0.0148 | -0.2755 +/- 0.1114 |
| milton_original | calibrated_logit_average | 0.7270 +/- 0.0159 | 0.7354 +/- 0.0154 | 0.5363 +/- 0.0700 | -0.1403 +/- 0.2251 |
| milton_original | calibrated_probability_average | 0.7283 +/- 0.0079 | 0.7362 +/- 0.0077 | 0.5389 +/- 0.0490 | -0.1366 +/- 0.2034 |
| milton_original | concat_reference | 0.7507 +/- 0.0388 | 0.7575 +/- 0.0377 | 0.6445 +/- 0.0806 | 0.1448 +/- 0.1799 |
| milton_original | crossview_reference | 0.7454 +/- 0.0127 | 0.7505 +/- 0.0135 | 0.6556 +/- 0.0258 | 0.1774 +/- 0.0946 |
| milton_original | raw_confidence_voting | 0.7165 +/- 0.0172 | 0.7249 +/- 0.0172 | 0.5010 +/- 0.0132 | -0.2371 +/- 0.0933 |
| milton_original | raw_logit_average | 0.7283 +/- 0.0079 | 0.7368 +/- 0.0079 | 0.5389 +/- 0.0490 | -0.1366 +/- 0.2034 |
| milton_original | raw_probability_average | 0.7283 +/- 0.0039 | 0.7364 +/- 0.0047 | 0.5393 +/- 0.0280 | -0.1367 +/- 0.1694 |
| milton_original | remote_only | 0.6824 +/- 0.0149 | 0.6900 +/- 0.0157 | 0.3759 +/- 0.0234 | -0.5737 +/- 0.1898 |
| milton_original | street_only | 0.7402 +/- 0.0205 | 0.7482 +/- 0.0191 | 0.5887 +/- 0.0372 | 0.0000 +/- 0.0000 |

## Paired comparison vs crossview on the conflict subset

Per-seed paired bootstrap CIs and exact McNemar p-values against the
crossview reference; `sig` counts seeds with p < 0.05.

| dataset | method | delta vs crossview | McNemar sig |
| --- | --- | --- | --- |
| altadena_3class | calibrated_confidence_voting | -0.0270 +/- 0.1215 | 2/3 |
| altadena_3class | calibrated_logit_average | -0.0246 +/- 0.1182 | 2/3 |
| altadena_3class | calibrated_probability_average | -0.0197 +/- 0.1218 | 2/3 |
| altadena_3class | concat_reference | -0.0051 +/- 0.0277 | 0/3 |
| altadena_3class | raw_confidence_voting | -0.0220 +/- 0.1248 | 2/3 |
| altadena_3class | raw_logit_average | -0.0296 +/- 0.1135 | 2/3 |
| altadena_3class | raw_probability_average | -0.0270 +/- 0.1213 | 2/3 |
| altadena_3class | remote_only | -0.2439 +/- 0.0986 | 3/3 |
| altadena_3class | street_only | -0.1943 +/- 0.1379 | 2/3 |
| ian_original | calibrated_confidence_voting | -0.0135 +/- 0.0403 | 0/3 |
| ian_original | calibrated_logit_average | -0.0070 +/- 0.0706 | 0/3 |
| ian_original | calibrated_probability_average | -0.0204 +/- 0.0769 | 1/3 |
| ian_original | concat_reference | -0.0299 +/- 0.0484 | 0/3 |
| ian_original | raw_confidence_voting | -0.0231 +/- 0.0481 | 0/3 |
| ian_original | raw_logit_average | -0.0044 +/- 0.0899 | 1/3 |
| ian_original | raw_probability_average | -0.0075 +/- 0.0788 | 0/3 |
| ian_original | remote_only | -0.2496 +/- 0.1075 | 2/3 |
| ian_original | street_only | -0.0643 +/- 0.0849 | 1/3 |
| milton_original | calibrated_confidence_voting | -0.1693 +/- 0.0122 | 2/3 |
| milton_original | calibrated_logit_average | -0.1193 +/- 0.0504 | 1/3 |
| milton_original | calibrated_probability_average | -0.1167 +/- 0.0412 | 0/3 |
| milton_original | concat_reference | -0.0112 +/- 0.0988 | 0/3 |
| milton_original | raw_confidence_voting | -0.1546 +/- 0.0336 | 1/3 |
| milton_original | raw_logit_average | -0.1167 +/- 0.0412 | 1/3 |
| milton_original | raw_probability_average | -0.1163 +/- 0.0343 | 0/3 |
| milton_original | remote_only | -0.2797 +/- 0.0465 | 3/3 |
| milton_original | street_only | -0.0669 +/- 0.0376 | 0/3 |
