# Reliability Gate Results (Phase 1)

Gate: per-sample street-vs-remote mixture weight predicted from building
visibility, calibrated confidence/entropy, confidence gap, JS divergence,
and a disagreement flag. Trained on val by minimizing the NLL of the gated
mixture; evaluated on test. `transfer` rows train the gate on a different
dataset's val split (zero-shot cross-disaster).

| dataset | variant | train source | accuracy | macro_f1 | conflict_acc | gap closure | sig vs crossview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| altadena_3class | gate3_linear_in_domain | altadena_3class | 0.9360 +/- 0.0031 | 0.7174 +/- 0.0154 | 0.7501 +/- 0.0151 | 0.5093 +/- 0.0492 | 1/3 |
| altadena_3class | gate3_mlp_in_domain | altadena_3class | 0.9336 +/- 0.0040 | 0.7165 +/- 0.0168 | 0.7262 +/- 0.0136 | 0.4510 +/- 0.0831 | 1/3 |
| altadena_3class | gate_linear_in_domain | altadena_3class | 0.9276 +/- 0.0117 | 0.7061 +/- 0.0185 | 0.7227 +/- 0.0174 | 0.4414 +/- 0.0415 | 1/3 |
| altadena_3class | gate_linear_loo_pooled | ian_original+milton_original | 0.9136 +/- 0.0171 | 0.7183 +/- 0.0177 | 0.6211 +/- 0.0625 | 0.1975 +/- 0.1151 | 2/3 |
| altadena_3class | gate_linear_loo_pooled_znorm | ian_original+milton_original | 0.9168 +/- 0.0139 | 0.7214 +/- 0.0161 | 0.6445 +/- 0.0331 | 0.2538 +/- 0.0642 | 3/3 |
| altadena_3class | gate_linear_transfer | ian_original | 0.9241 +/- 0.0179 | 0.7137 +/- 0.0201 | 0.6976 +/- 0.0639 | 0.3787 +/- 0.1289 | 2/3 |
| altadena_3class | gate_linear_transfer | milton_original | 0.8975 +/- 0.0182 | 0.7041 +/- 0.0148 | 0.5045 +/- 0.0604 | -0.0834 +/- 0.1469 | 2/3 |
| altadena_3class | gate_mlp_in_domain | altadena_3class | 0.9278 +/- 0.0102 | 0.7095 +/- 0.0114 | 0.7242 +/- 0.0188 | 0.4449 +/- 0.0974 | 1/3 |
| altadena_3class | gate_mlp_loo_pooled | ian_original+milton_original | 0.9000 +/- 0.0178 | 0.7076 +/- 0.0163 | 0.5215 +/- 0.0751 | -0.0395 +/- 0.1376 | 2/3 |
| altadena_3class | gate_mlp_loo_pooled_znorm | ian_original+milton_original | 0.9066 +/- 0.0166 | 0.7107 +/- 0.0153 | 0.5701 +/- 0.0527 | 0.0748 +/- 0.1092 | 2/3 |
| altadena_3class | gate_mlp_transfer | ian_original | 0.9051 +/- 0.0100 | 0.7083 +/- 0.0103 | 0.5576 +/- 0.0411 | 0.0498 +/- 0.0415 | 2/3 |
| altadena_3class | gate_mlp_transfer | milton_original | 0.8923 +/- 0.0198 | 0.6966 +/- 0.0121 | 0.4663 +/- 0.0757 | -0.1744 +/- 0.1680 | 2/3 |
| ian_original | gate3_linear_in_domain | ian_original | 0.7344 +/- 0.0150 | 0.7377 +/- 0.0155 | 0.6134 +/- 0.0590 | 0.1770 +/- 0.1555 | 0/3 |
| ian_original | gate3_mlp_in_domain | ian_original | 0.7156 +/- 0.0168 | 0.7198 +/- 0.0164 | 0.5723 +/- 0.0687 | 0.0702 +/- 0.1215 | 0/3 |
| ian_original | gate_linear_in_domain | ian_original | 0.7289 +/- 0.0102 | 0.7328 +/- 0.0110 | 0.6044 +/- 0.0817 | 0.1561 +/- 0.1846 | 0/3 |
| ian_original | gate_linear_loo_pooled | altadena_3class+milton_original | 0.7233 +/- 0.0120 | 0.7276 +/- 0.0105 | 0.5872 +/- 0.0241 | 0.1068 +/- 0.0648 | 0/3 |
| ian_original | gate_linear_loo_pooled_znorm | altadena_3class+milton_original | 0.7244 +/- 0.0158 | 0.7291 +/- 0.0146 | 0.5903 +/- 0.0226 | 0.1166 +/- 0.0564 | 1/3 |
| ian_original | gate_linear_transfer | altadena_3class | 0.7233 +/- 0.0120 | 0.7276 +/- 0.0105 | 0.5877 +/- 0.0422 | 0.1142 +/- 0.0160 | 0/3 |
| ian_original | gate_linear_transfer | milton_original | 0.7089 +/- 0.0038 | 0.7132 +/- 0.0032 | 0.5463 +/- 0.0533 | -0.0057 +/- 0.1036 | 0/3 |
| ian_original | gate_mlp_in_domain | ian_original | 0.7167 +/- 0.0120 | 0.7202 +/- 0.0122 | 0.5694 +/- 0.0859 | 0.0618 +/- 0.1745 | 0/3 |
| ian_original | gate_mlp_loo_pooled | altadena_3class+milton_original | 0.7111 +/- 0.0107 | 0.7147 +/- 0.0102 | 0.5529 +/- 0.0583 | 0.0189 +/- 0.0532 | 1/3 |
| ian_original | gate_mlp_loo_pooled_znorm | altadena_3class+milton_original | 0.6856 +/- 0.0051 | 0.6887 +/- 0.0041 | 0.4791 +/- 0.0599 | -0.1953 +/- 0.1404 | 1/3 |
| ian_original | gate_mlp_transfer | altadena_3class | 0.7233 +/- 0.0100 | 0.7268 +/- 0.0077 | 0.5872 +/- 0.0248 | 0.1045 +/- 0.0905 | 0/3 |
| ian_original | gate_mlp_transfer | milton_original | 0.7022 +/- 0.0069 | 0.7070 +/- 0.0077 | 0.5269 +/- 0.0659 | -0.0673 +/- 0.1991 | 0/3 |
| milton_original | gate3_linear_in_domain | milton_original | 0.7559 +/- 0.0079 | 0.7631 +/- 0.0078 | 0.6564 +/- 0.0321 | 0.1773 +/- 0.0721 | 0/3 |
| milton_original | gate3_mlp_in_domain | milton_original | 0.7572 +/- 0.0159 | 0.7653 +/- 0.0139 | 0.6493 +/- 0.0389 | 0.1607 +/- 0.0274 | 0/3 |
| milton_original | gate_linear_in_domain | milton_original | 0.7428 +/- 0.0091 | 0.7509 +/- 0.0081 | 0.5938 +/- 0.0059 | 0.0105 +/- 0.0929 | 0/3 |
| milton_original | gate_linear_loo_pooled | altadena_3class+ian_original | 0.7297 +/- 0.0186 | 0.7379 +/- 0.0181 | 0.5489 +/- 0.0252 | -0.1087 +/- 0.0714 | 0/3 |
| milton_original | gate_linear_loo_pooled_znorm | altadena_3class+ian_original | 0.7310 +/- 0.0231 | 0.7392 +/- 0.0222 | 0.5555 +/- 0.0395 | -0.0899 +/- 0.0522 | 0/3 |
| milton_original | gate_linear_transfer | altadena_3class | 0.7244 +/- 0.0180 | 0.7325 +/- 0.0174 | 0.5310 +/- 0.0309 | -0.1540 +/- 0.0719 | 0/3 |
| milton_original | gate_linear_transfer | ian_original | 0.7323 +/- 0.0180 | 0.7402 +/- 0.0170 | 0.5597 +/- 0.0352 | -0.0770 +/- 0.0672 | 0/3 |
| milton_original | gate_mlp_in_domain | milton_original | 0.7441 +/- 0.0104 | 0.7525 +/- 0.0095 | 0.5963 +/- 0.0265 | 0.0142 +/- 0.1465 | 0/3 |
| milton_original | gate_mlp_loo_pooled | altadena_3class+ian_original | 0.7257 +/- 0.0149 | 0.7346 +/- 0.0135 | 0.5342 +/- 0.0140 | -0.1472 +/- 0.0765 | 0/3 |
| milton_original | gate_mlp_loo_pooled_znorm | altadena_3class+ian_original | 0.7336 +/- 0.0149 | 0.7424 +/- 0.0143 | 0.5629 +/- 0.0187 | -0.0702 +/- 0.0616 | 0/3 |
| milton_original | gate_mlp_transfer | altadena_3class | 0.7205 +/- 0.0341 | 0.7284 +/- 0.0338 | 0.5225 +/- 0.0699 | -0.1743 +/- 0.0795 | 1/3 |
| milton_original | gate_mlp_transfer | ian_original | 0.7415 +/- 0.0217 | 0.7500 +/- 0.0210 | 0.5936 +/- 0.0408 | 0.0128 +/- 0.0222 | 0/3 |

## Linear gate coefficients (in-domain, positive = trust street view)

| dataset | feature | coefficient |
| --- | --- | --- |
| altadena_3class | building_ratio | -0.6724 +/- 0.1175 |
| altadena_3class | center_building_ratio | -0.3170 +/- 0.4503 |
| altadena_3class | center_minus_global | 0.7893 +/- 0.1818 |
| altadena_3class | centroid_distance_norm | 0.0102 +/- 0.1912 |
| altadena_3class | confidence_gap | 0.3966 +/- 0.1606 |
| altadena_3class | js_divergence | -0.0739 +/- 0.6019 |
| altadena_3class | remote_confidence | -0.2964 +/- 0.1960 |
| altadena_3class | remote_entropy | 0.4053 +/- 0.2506 |
| altadena_3class | street_confidence | 0.3536 +/- 0.1795 |
| altadena_3class | street_entropy | -0.8263 +/- 0.3649 |
| altadena_3class | views_disagree | 0.6043 +/- 0.7137 |
| ian_original | building_ratio | -0.2159 +/- 0.1350 |
| ian_original | center_building_ratio | -0.0696 +/- 0.2247 |
| ian_original | center_minus_global | 0.2821 +/- 0.3631 |
| ian_original | centroid_distance_norm | -0.2888 +/- 0.5767 |
| ian_original | confidence_gap | 0.3805 +/- 0.1535 |
| ian_original | js_divergence | 0.2628 +/- 0.6244 |
| ian_original | remote_confidence | -0.3615 +/- 0.5615 |
| ian_original | remote_entropy | 0.5864 +/- 0.2391 |
| ian_original | street_confidence | 0.2388 +/- 0.1414 |
| ian_original | street_entropy | -0.4068 +/- 0.2665 |
| ian_original | views_disagree | -0.1653 +/- 0.7213 |
| milton_original | building_ratio | 0.7865 +/- 0.5947 |
| milton_original | center_building_ratio | 0.4631 +/- 0.2882 |
| milton_original | center_minus_global | -0.2889 +/- 1.0434 |
| milton_original | centroid_distance_norm | -0.2826 +/- 0.1119 |
| milton_original | confidence_gap | 0.0769 +/- 0.6199 |
| milton_original | js_divergence | 0.4566 +/- 0.5014 |
| milton_original | remote_confidence | -0.1625 +/- 0.6480 |
| milton_original | remote_entropy | -0.2900 +/- 1.1005 |
| milton_original | street_confidence | -0.4216 +/- 0.4831 |
| milton_original | street_entropy | 0.1047 +/- 0.6073 |
| milton_original | views_disagree | -0.5515 +/- 0.6657 |
