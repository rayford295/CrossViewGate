# Pooled-Across-Seeds Paired Tests

One test per comparison instead of 'significant in k/N seeds'. Unit of
analysis is the test sample; its paired correctness difference is averaged
over the seeds in which it qualifies (conflict scope: seeds where the two
single-view models disagree on it). H0: E[delta] = 0, two-sided sign-flip
permutation test; percentile bootstrap 95% CI.

| dataset | comparison | scope | n | mean delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| altadena_3class | crossview - street_only | conflict | 489 | +0.2515 | [+0.1953, +0.3067] | 0.0000 |
| altadena_3class | crossview - remote_only | conflict | 489 | +0.2829 | [+0.2335, +0.3327] | 0.0000 |
| altadena_3class | crossview - concat | conflict | 489 | +0.0034 | [-0.0235, +0.0300] | 0.8080 |
| altadena_3class | crossview - prob_average | conflict | 489 | +0.0436 | [+0.0119, +0.0763] | 0.0087 |
| altadena_3class | gate3_linear - crossview | conflict | 489 | +0.0429 | [+0.0187, +0.0668] | 0.0006 |
| altadena_3class | gate3_linear - crossview | full | 1984 | +0.0099 | [+0.0055, +0.0144] | 0.0000 |
| altadena_3class | gate3_linear - prob_average | conflict | 489 | +0.0866 | [+0.0569, +0.1172] | 0.0000 |
| altadena_3class | gate_linear - crossview | conflict | 489 | +0.0099 | [-0.0242, +0.0440] | 0.5832 |
| altadena_3class | crossview - street_only | full | 1984 | +0.0286 | [+0.0190, +0.0380] | 0.0000 |
| altadena_3class | crossview - remote_only | full | 1984 | +0.0356 | [+0.0277, +0.0433] | 0.0000 |
| ian_original | crossview - street_only | conflict | 179 | +0.0987 | [+0.0214, +0.1778] | 0.0151 |
| ian_original | crossview - remote_only | conflict | 179 | +0.2458 | [+0.1443, +0.3492] | 0.0000 |
| ian_original | crossview - concat | conflict | 179 | +0.0335 | [-0.0233, +0.0912] | 0.2621 |
| ian_original | crossview - prob_average | conflict | 179 | +0.0140 | [-0.0438, +0.0717] | 0.6485 |
| ian_original | gate3_linear - crossview | conflict | 179 | -0.0056 | [-0.0373, +0.0242] | 0.7663 |
| ian_original | gate3_linear - crossview | full | 300 | +0.0133 | [-0.0022, +0.0289] | 0.0894 |
| ian_original | gate3_linear - prob_average | conflict | 179 | +0.0084 | [-0.0484, +0.0652] | 0.7821 |
| ian_original | gate_linear - crossview | conflict | 179 | -0.0317 | [-0.0847, +0.0186] | 0.2530 |
| ian_original | crossview - street_only | full | 300 | +0.0122 | [-0.0156, +0.0400] | 0.3837 |
| ian_original | crossview - remote_only | full | 300 | +0.0756 | [+0.0322, +0.1189] | 0.0009 |
| milton_original | crossview - street_only | conflict | 128 | +0.1276 | [+0.0117, +0.2409] | 0.0334 |
| milton_original | crossview - remote_only | conflict | 128 | +0.2721 | [+0.1667, +0.3737] | 0.0000 |
| milton_original | crossview - concat | conflict | 128 | +0.0456 | [-0.0299, +0.1224] | 0.2609 |
| milton_original | crossview - prob_average | conflict | 128 | +0.1029 | [+0.0260, +0.1797] | 0.0125 |
| milton_original | gate3_linear - crossview | conflict | 128 | -0.0195 | [-0.1042, +0.0625] | 0.6591 |
| milton_original | gate3_linear - crossview | full | 254 | +0.0105 | [-0.0144, +0.0367] | 0.4075 |
| milton_original | gate3_linear - prob_average | conflict | 128 | +0.0833 | [-0.0130, +0.1784] | 0.0999 |
| milton_original | gate_linear - crossview | conflict | 128 | -0.0898 | [-0.1862, +0.0078] | 0.0760 |
| milton_original | crossview - street_only | full | 254 | +0.0052 | [-0.0289, +0.0394] | 0.8090 |
| milton_original | crossview - remote_only | full | 254 | +0.0630 | [+0.0262, +0.0997] | 0.0012 |
