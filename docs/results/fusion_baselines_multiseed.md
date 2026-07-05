# Fusion Baseline Multiseed Results

Mean +/- std over completed seeds.

| dataset | method | accuracy | macro_f1 | weighted_f1 | conflict_acc |
| --- | --- | --- | --- | --- | --- |
| altadena_3class | concat_reference | 0.9153 +/- 0.0189 | 0.7051 +/- 0.0097 | 0.9345 +/- 0.0107 | 0.6920 +/- 0.0813 |
| altadena_3class | confidence_voting | 0.9209 +/- 0.0142 | 0.7044 +/- 0.0121 | 0.9358 +/- 0.0079 | 0.6751 +/- 0.0519 |
| altadena_3class | crossview_reference | 0.9261 +/- 0.0128 | 0.7165 +/- 0.0037 | 0.9408 +/- 0.0062 | 0.6971 +/- 0.0895 |
| altadena_3class | late_fusion_logit_average | 0.9199 +/- 0.0127 | 0.7096 +/- 0.0143 | 0.9370 +/- 0.0078 | 0.6675 +/- 0.0491 |
| altadena_3class | late_fusion_probability_average | 0.9202 +/- 0.0138 | 0.7053 +/- 0.0129 | 0.9359 +/- 0.0078 | 0.6702 +/- 0.0536 |
| altadena_3class | remote_only | 0.8905 +/- 0.0122 | 0.6729 +/- 0.0096 | 0.9159 +/- 0.0068 | 0.4532 +/- 0.0595 |
| altadena_3class | street_only | 0.8975 +/- 0.0178 | 0.7079 +/- 0.0166 | 0.9214 +/- 0.0112 | 0.5028 +/- 0.0798 |
| ian_original | concat_reference | 0.6878 +/- 0.0069 | 0.6837 +/- 0.0081 | 0.6837 +/- 0.0081 | 0.5805 +/- 0.0407 |
| ian_original | confidence_voting | 0.7233 +/- 0.0058 | 0.7276 +/- 0.0059 | 0.7276 +/- 0.0059 | 0.5874 +/- 0.0386 |
| ian_original | crossview_reference | 0.7211 +/- 0.0379 | 0.7240 +/- 0.0371 | 0.7240 +/- 0.0371 | 0.6104 +/- 0.0854 |
| ian_original | late_fusion_logit_average | 0.7300 +/- 0.0186 | 0.7347 +/- 0.0177 | 0.7347 +/- 0.0177 | 0.6060 +/- 0.0154 |
| ian_original | late_fusion_probability_average | 0.7289 +/- 0.0150 | 0.7332 +/- 0.0140 | 0.7332 +/- 0.0140 | 0.6030 +/- 0.0140 |
| ian_original | remote_only | 0.6456 +/- 0.0302 | 0.6437 +/- 0.0331 | 0.6437 +/- 0.0331 | 0.3608 +/- 0.0588 |
| ian_original | street_only | 0.7089 +/- 0.0150 | 0.7129 +/- 0.0138 | 0.7129 +/- 0.0138 | 0.5462 +/- 0.0494 |
| milton_original | concat_reference | 0.7507 +/- 0.0388 | 0.7575 +/- 0.0377 | 0.7505 +/- 0.0396 | 0.6445 +/- 0.0806 |
| milton_original | confidence_voting | 0.7165 +/- 0.0172 | 0.7249 +/- 0.0172 | 0.7118 +/- 0.0186 | 0.5010 +/- 0.0132 |
| milton_original | crossview_reference | 0.7454 +/- 0.0127 | 0.7505 +/- 0.0135 | 0.7434 +/- 0.0143 | 0.6556 +/- 0.0258 |
| milton_original | late_fusion_logit_average | 0.7283 +/- 0.0079 | 0.7368 +/- 0.0079 | 0.7273 +/- 0.0079 | 0.5389 +/- 0.0490 |
| milton_original | late_fusion_probability_average | 0.7283 +/- 0.0039 | 0.7364 +/- 0.0047 | 0.7263 +/- 0.0047 | 0.5393 +/- 0.0280 |
| milton_original | remote_only | 0.6824 +/- 0.0149 | 0.6900 +/- 0.0157 | 0.6793 +/- 0.0171 | 0.3759 +/- 0.0234 |
| milton_original | street_only | 0.7402 +/- 0.0205 | 0.7482 +/- 0.0191 | 0.7397 +/- 0.0212 | 0.5887 +/- 0.0372 |
