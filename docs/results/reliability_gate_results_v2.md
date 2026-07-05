# Reliability Gate Results (Phase 1)

Gate: per-sample street-vs-remote mixture weight predicted from building
visibility, calibrated confidence/entropy, confidence gap, JS divergence,
and a disagreement flag. Trained on val by minimizing the NLL of the gated
mixture; evaluated on test. `transfer` rows train the gate on a different
dataset's val split (zero-shot cross-disaster).

| dataset | variant | train source | accuracy | macro_f1 | conflict_acc | gap closure | sig vs crossview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| altadena_3class | gate3_linear_in_domain | altadena_3class | 0.9588 +/- 0.0085 | 0.7342 +/- 0.0097 | 0.7678 +/- 0.0312 | 0.5237 +/- 0.0930 | 2/5 |
| altadena_3class | gate3_mlp_in_domain | altadena_3class | 0.9530 +/- 0.0127 | 0.7221 +/- 0.0103 | 0.7193 +/- 0.0346 | 0.4021 +/- 0.1248 | 1/5 |
| altadena_3class | gate_linear_in_domain | altadena_3class | 0.9546 +/- 0.0106 | 0.7130 +/- 0.0162 | 0.7343 +/- 0.0280 | 0.4425 +/- 0.0947 | 2/5 |
| altadena_3class | gate_linear_loo_pooled | ian_original+milton_original | 0.9444 +/- 0.0175 | 0.7165 +/- 0.0070 | 0.6325 +/- 0.0580 | 0.1914 +/- 0.1609 | 2/5 |
| altadena_3class | gate_linear_loo_pooled_znorm | ian_original+milton_original | 0.9474 +/- 0.0157 | 0.7145 +/- 0.0104 | 0.6650 +/- 0.0383 | 0.2725 +/- 0.1137 | 1/5 |
| altadena_3class | gate_linear_transfer | ian_original | 0.9483 +/- 0.0136 | 0.7234 +/- 0.0117 | 0.6669 +/- 0.0486 | 0.2727 +/- 0.1530 | 0/5 |
| altadena_3class | gate_linear_transfer | milton_original | 0.9372 +/- 0.0202 | 0.7133 +/- 0.0109 | 0.5579 +/- 0.0773 | 0.0119 +/- 0.1796 | 3/5 |
| altadena_3class | gate_mlp_in_domain | altadena_3class | 0.9501 +/- 0.0121 | 0.7161 +/- 0.0172 | 0.6844 +/- 0.0479 | 0.3199 +/- 0.1122 | 1/5 |
| altadena_3class | gate_mlp_loo_pooled | ian_original+milton_original | 0.9378 +/- 0.0150 | 0.7172 +/- 0.0023 | 0.5553 +/- 0.0663 | -0.0077 +/- 0.2605 | 3/5 |
| altadena_3class | gate_mlp_loo_pooled_znorm | ian_original+milton_original | 0.9352 +/- 0.0182 | 0.7186 +/- 0.0046 | 0.5340 +/- 0.0776 | -0.0457 +/- 0.1930 | 3/5 |
| altadena_3class | gate_mlp_transfer | ian_original | 0.9359 +/- 0.0175 | 0.7106 +/- 0.0104 | 0.5407 +/- 0.0763 | -0.0433 +/- 0.2871 | 3/5 |
| altadena_3class | gate_mlp_transfer | milton_original | 0.9357 +/- 0.0207 | 0.7167 +/- 0.0151 | 0.5465 +/- 0.0416 | -0.0247 +/- 0.1885 | 4/5 |
| ian_original | gate3_linear_in_domain | ian_original | 0.7307 +/- 0.0104 | 0.7344 +/- 0.0109 | 0.6180 +/- 0.0366 | 0.2338 +/- 0.1620 | 0/5 |
| ian_original | gate3_mlp_in_domain | ian_original | 0.7253 +/- 0.0168 | 0.7290 +/- 0.0174 | 0.6013 +/- 0.0388 | 0.1954 +/- 0.0695 | 0/5 |
| ian_original | gate_linear_in_domain | ian_original | 0.7333 +/- 0.0127 | 0.7361 +/- 0.0131 | 0.6221 +/- 0.0222 | 0.2477 +/- 0.0952 | 0/5 |
| ian_original | gate_linear_loo_pooled | altadena_3class+milton_original | 0.7293 +/- 0.0086 | 0.7321 +/- 0.0095 | 0.6106 +/- 0.0128 | 0.2122 +/- 0.1044 | 0/5 |
| ian_original | gate_linear_loo_pooled_znorm | altadena_3class+milton_original | 0.7273 +/- 0.0128 | 0.7300 +/- 0.0135 | 0.6043 +/- 0.0202 | 0.1981 +/- 0.0845 | 0/5 |
| ian_original | gate_linear_transfer | altadena_3class | 0.7200 +/- 0.0078 | 0.7228 +/- 0.0080 | 0.5824 +/- 0.0314 | 0.1317 +/- 0.1718 | 0/5 |
| ian_original | gate_linear_transfer | milton_original | 0.7107 +/- 0.0157 | 0.7138 +/- 0.0169 | 0.5538 +/- 0.0404 | 0.0628 +/- 0.0727 | 1/5 |
| ian_original | gate_mlp_in_domain | ian_original | 0.6940 +/- 0.0157 | 0.6964 +/- 0.0162 | 0.5037 +/- 0.0341 | -0.0758 +/- 0.0729 | 2/5 |
| ian_original | gate_mlp_loo_pooled | altadena_3class+milton_original | 0.7020 +/- 0.0145 | 0.7044 +/- 0.0143 | 0.5271 +/- 0.0357 | -0.0108 +/- 0.0772 | 1/5 |
| ian_original | gate_mlp_loo_pooled_znorm | altadena_3class+milton_original | 0.7060 +/- 0.0080 | 0.7079 +/- 0.0095 | 0.5400 +/- 0.0243 | 0.0188 +/- 0.1202 | 1/5 |
| ian_original | gate_mlp_transfer | altadena_3class | 0.7073 +/- 0.0169 | 0.7095 +/- 0.0173 | 0.5449 +/- 0.0559 | 0.0296 +/- 0.1924 | 1/5 |
| ian_original | gate_mlp_transfer | milton_original | 0.7027 +/- 0.0192 | 0.7049 +/- 0.0206 | 0.5291 +/- 0.0470 | -0.0012 +/- 0.0793 | 1/5 |
| milton_original | gate3_linear_in_domain | milton_original | 0.7772 +/- 0.0154 | 0.7820 +/- 0.0146 | 0.6309 +/- 0.0525 | 0.1813 +/- 0.0713 | 0/5 |
| milton_original | gate3_mlp_in_domain | milton_original | 0.7835 +/- 0.0096 | 0.7883 +/- 0.0090 | 0.6643 +/- 0.0420 | 0.2641 +/- 0.0448 | 0/5 |
| milton_original | gate_linear_in_domain | milton_original | 0.7646 +/- 0.0232 | 0.7699 +/- 0.0229 | 0.5793 +/- 0.0737 | 0.0598 +/- 0.1217 | 0/5 |
| milton_original | gate_linear_loo_pooled | altadena_3class+ian_original | 0.7756 +/- 0.0124 | 0.7812 +/- 0.0116 | 0.6219 +/- 0.0274 | 0.1550 +/- 0.1165 | 0/5 |
| milton_original | gate_linear_loo_pooled_znorm | altadena_3class+ian_original | 0.7764 +/- 0.0146 | 0.7819 +/- 0.0133 | 0.6255 +/- 0.0419 | 0.1645 +/- 0.0999 | 0/5 |
| milton_original | gate_linear_transfer | altadena_3class | 0.7677 +/- 0.0134 | 0.7730 +/- 0.0124 | 0.5915 +/- 0.0462 | 0.0763 +/- 0.1419 | 0/5 |
| milton_original | gate_linear_transfer | ian_original | 0.7748 +/- 0.0170 | 0.7802 +/- 0.0157 | 0.6191 +/- 0.0442 | 0.1511 +/- 0.0751 | 0/5 |
| milton_original | gate_mlp_in_domain | milton_original | 0.7701 +/- 0.0180 | 0.7753 +/- 0.0176 | 0.6002 +/- 0.0589 | 0.1043 +/- 0.1078 | 0/5 |
| milton_original | gate_mlp_loo_pooled | altadena_3class+ian_original | 0.7677 +/- 0.0199 | 0.7739 +/- 0.0191 | 0.5879 +/- 0.0691 | 0.0695 +/- 0.2153 | 1/5 |
| milton_original | gate_mlp_loo_pooled_znorm | altadena_3class+ian_original | 0.7567 +/- 0.0205 | 0.7621 +/- 0.0190 | 0.5471 +/- 0.0575 | -0.0347 +/- 0.2025 | 0/5 |
| milton_original | gate_mlp_transfer | altadena_3class | 0.7551 +/- 0.0170 | 0.7609 +/- 0.0162 | 0.5420 +/- 0.0355 | -0.0415 +/- 0.0855 | 0/5 |
| milton_original | gate_mlp_transfer | ian_original | 0.7520 +/- 0.0183 | 0.7585 +/- 0.0177 | 0.5292 +/- 0.0404 | -0.0739 +/- 0.1740 | 1/5 |

## Linear gate coefficients (in-domain, positive = trust street view)

| dataset | feature | coefficient |
| --- | --- | --- |
| altadena_3class | building_ratio | -0.4461 +/- 0.3050 |
| altadena_3class | center_building_ratio | -0.1323 +/- 0.3194 |
| altadena_3class | center_minus_global | 0.7546 +/- 0.3304 |
| altadena_3class | centroid_distance_norm | 0.2329 +/- 0.1859 |
| altadena_3class | confidence_gap | 0.3390 +/- 0.1594 |
| altadena_3class | js_divergence | -0.2357 +/- 0.3601 |
| altadena_3class | remote_confidence | -0.1730 +/- 0.2515 |
| altadena_3class | remote_entropy | 0.2302 +/- 0.0881 |
| altadena_3class | street_confidence | 0.3721 +/- 0.2757 |
| altadena_3class | street_entropy | -0.4690 +/- 0.5661 |
| altadena_3class | views_disagree | 0.3849 +/- 0.6306 |
| ian_original | building_ratio | -0.0069 +/- 0.1578 |
| ian_original | center_building_ratio | -0.0769 +/- 0.1987 |
| ian_original | center_minus_global | 0.0214 +/- 0.1643 |
| ian_original | centroid_distance_norm | -0.3575 +/- 0.2312 |
| ian_original | confidence_gap | 0.2454 +/- 0.2256 |
| ian_original | js_divergence | 0.1890 +/- 0.2293 |
| ian_original | remote_confidence | -0.3286 +/- 0.6183 |
| ian_original | remote_entropy | 0.1919 +/- 0.6136 |
| ian_original | street_confidence | -0.1002 +/- 0.4937 |
| ian_original | street_entropy | -0.4808 +/- 0.4721 |
| ian_original | views_disagree | -0.3346 +/- 0.3749 |
| milton_original | building_ratio | 0.4171 +/- 0.5470 |
| milton_original | center_building_ratio | 0.0763 +/- 0.2518 |
| milton_original | center_minus_global | -0.1035 +/- 0.2978 |
| milton_original | centroid_distance_norm | -0.2414 +/- 0.1317 |
| milton_original | confidence_gap | -0.0549 +/- 0.7178 |
| milton_original | js_divergence | -0.4437 +/- 0.6613 |
| milton_original | remote_confidence | -0.5871 +/- 1.0503 |
| milton_original | remote_entropy | 0.1743 +/- 1.6525 |
| milton_original | street_confidence | -1.1361 +/- 1.1731 |
| milton_original | street_entropy | -1.3206 +/- 0.9601 |
| milton_original | views_disagree | 0.0882 +/- 0.7568 |
