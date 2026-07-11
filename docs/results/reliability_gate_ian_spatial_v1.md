# Reliability Gate Results (Phase 1)

Gate: per-sample street-vs-remote mixture weight predicted from building
visibility, calibrated confidence/entropy, confidence gap, JS divergence,
and a disagreement flag. Trained on gate-fit (legacy runs use full val) by
minimizing the NLL of the gated mixture; evaluated on final test. `transfer`
rows train the gate on a different dataset's gate-fit split.

| dataset | variant | train source | accuracy | macro_f1 | conflict_acc | gap closure | sig vs crossview |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ian_original | gate3_linear_in_domain | ian_original | 0.6472 +/- 0.0147 | 0.6501 +/- 0.0139 | 0.5117 +/- 0.0413 | 0.0433 +/- 0.0744 | 1/5 |
| ian_original | gate3_mlp_in_domain | ian_original | 0.6448 +/- 0.0169 | 0.6483 +/- 0.0168 | 0.4938 +/- 0.0411 | -0.0087 +/- 0.1265 | 0/5 |
| ian_original | gate_linear_in_domain | ian_original | 0.6501 +/- 0.0159 | 0.6526 +/- 0.0161 | 0.5043 +/- 0.0394 | 0.0174 +/- 0.1553 | 1/5 |
| ian_original | gate_mlp_in_domain | ian_original | 0.6410 +/- 0.0222 | 0.6443 +/- 0.0226 | 0.4846 +/- 0.0549 | -0.0390 +/- 0.2027 | 1/5 |

## Linear gate coefficients (in-domain, positive = trust street view)

| dataset | feature | coefficient |
| --- | --- | --- |
| ian_original | building_ratio | 0.0816 +/- 0.6305 |
| ian_original | center_building_ratio | 0.5082 +/- 0.4686 |
| ian_original | center_minus_global | 1.1061 +/- 0.3718 |
| ian_original | centroid_distance_norm | -0.6110 +/- 0.5967 |
| ian_original | confidence_gap | 0.5498 +/- 0.2623 |
| ian_original | js_divergence | -0.2737 +/- 0.7751 |
| ian_original | remote_confidence | -0.7834 +/- 1.3193 |
| ian_original | remote_entropy | 0.0504 +/- 0.9869 |
| ian_original | street_confidence | -0.0620 +/- 1.1129 |
| ian_original | street_entropy | -0.4598 +/- 0.5436 |
| ian_original | views_disagree | -0.6748 +/- 0.7087 |
