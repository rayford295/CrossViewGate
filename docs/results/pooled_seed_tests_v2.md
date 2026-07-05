# Pooled-Across-Seeds Paired Tests

One test per comparison instead of 'significant in k/N seeds'. Unit of
analysis is the test sample; its paired correctness difference is averaged
over the seeds in which it qualifies (conflict scope: seeds where the two
single-view models disagree on it). H0: E[delta] = 0, two-sided sign-flip
permutation test; percentile bootstrap 95% CI.

| dataset | comparison | scope | n | mean delta | 95% CI | p |
| --- | --- | --- | --- | --- | --- | --- |
| altadena_3class | crossview - street_only | conflict | 451 | +0.2455 | [+0.1903, +0.3004] | 0.0000 |
| altadena_3class | crossview - remote_only | conflict | 451 | +0.3033 | [+0.2512, +0.3556] | 0.0000 |
| altadena_3class | crossview - concat | conflict | 451 | +0.0169 | [-0.0126, +0.0465] | 0.2734 |
| altadena_3class | crossview - prob_average | conflict | 451 | +0.0340 | [-0.0029, +0.0708] | 0.0729 |
| altadena_3class | gate3_linear - crossview | conflict | 451 | +0.0716 | [+0.0458, +0.0974] | 0.0000 |
| altadena_3class | gate3_linear - crossview | full | 1984 | +0.0184 | [+0.0143, +0.0228] | 0.0000 |
| altadena_3class | gate3_linear - prob_average | conflict | 451 | +0.1056 | [+0.0744, +0.1374] | 0.0000 |
| altadena_3class | gate_linear - crossview | conflict | 451 | +0.0340 | [+0.0007, +0.0666] | 0.0465 |
| altadena_3class | crossview - street_only | full | 1984 | +0.0090 | [+0.0020, +0.0159] | 0.0133 |
| altadena_3class | crossview - remote_only | full | 1984 | +0.0111 | [+0.0048, +0.0172] | 0.0001 |
| ian_original | crossview - street_only | conflict | 193 | +0.0777 | [+0.0179, +0.1392] | 0.0126 |
| ian_original | crossview - remote_only | conflict | 193 | +0.2877 | [+0.1878, +0.3817] | 0.0000 |
| ian_original | crossview - concat | conflict | 193 | +0.0136 | [-0.0313, +0.0584] | 0.5631 |
| ian_original | crossview - prob_average | conflict | 193 | -0.0031 | [-0.0542, +0.0486] | 0.9047 |
| ian_original | gate3_linear - crossview | conflict | 193 | +0.0054 | [-0.0207, +0.0325] | 0.6968 |
| ian_original | gate3_linear - crossview | full | 300 | -0.0040 | [-0.0180, +0.0100] | 0.5868 |
| ian_original | gate3_linear - prob_average | conflict | 193 | +0.0023 | [-0.0465, +0.0504] | 0.9297 |
| ian_original | gate_linear - crossview | conflict | 193 | +0.0257 | [-0.0317, +0.0813] | 0.3791 |
| ian_original | crossview - street_only | full | 300 | +0.0320 | [+0.0107, +0.0533] | 0.0045 |
| ian_original | crossview - remote_only | full | 300 | +0.0867 | [+0.0480, +0.1247] | 0.0000 |
| milton_original | crossview - street_only | conflict | 140 | +0.1407 | [+0.0445, +0.2396] | 0.0056 |
| milton_original | crossview - remote_only | conflict | 140 | +0.2099 | [+0.1043, +0.3131] | 0.0001 |
| milton_original | crossview - concat | conflict | 140 | +0.0049 | [-0.0660, +0.0730] | 0.8954 |
| milton_original | crossview - prob_average | conflict | 140 | +0.0385 | [-0.0345, +0.1111] | 0.2992 |
| milton_original | gate3_linear - crossview | conflict | 140 | -0.0149 | [-0.0676, +0.0364] | 0.5737 |
| milton_original | gate3_linear - crossview | full | 254 | +0.0008 | [-0.0150, +0.0157] | 0.8428 |
| milton_original | gate3_linear - prob_average | conflict | 140 | +0.0236 | [-0.0452, +0.0929] | 0.5011 |
| milton_original | gate_linear - crossview | conflict | 140 | -0.0773 | [-0.1652, +0.0117] | 0.0857 |
| milton_original | crossview - street_only | full | 254 | +0.0173 | [-0.0071, +0.0417] | 0.1560 |
| milton_original | crossview - remote_only | full | 254 | +0.0559 | [+0.0189, +0.0953] | 0.0038 |
