# Error analysis (test set, tuned threshold)

Counts: {'TN': 13122, 'FP': 3554, 'TP': 1614, 'FN': 165}

Median feature values by outcome:

| kind   |   absolute_magnitude |   relative_velocity |   miss_distance |
|:-------|---------------------:|--------------------:|----------------:|
| FN     |                20.3  |             41458.9 |     5.45e+07    |
| FP     |                20.4  |             61142.1 |     4.80847e+07 |
| TN     |                24.8  |             37405.2 |     3.2958e+07  |
| TP     |                20.47 |             61214.2 |     3.88921e+07 |


Performance by absolute-magnitude bin:

| H_bin    |   rows |   pha_share |   recall |   false_alarm_rate |
|:---------|-------:|------------:|---------:|-------------------:|
| (0, 18]  |    490 |       0.296 |    0.828 |              0.754 |
| (18, 20] |   1880 |       0.267 |    0.886 |              0.819 |
| (20, 22] |   3629 |       0.309 |    0.934 |              0.858 |
| (22, 24] |   4033 |       0.002 |    0.1   |              0.004 |
| (24, 26] |   4585 |       0     |  nan     |              0     |
| (26, 40] |   3838 |       0     |  nan     |              0     |

