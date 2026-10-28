# Late-window nRMSE slopes

Ordinary least-squares slope over steps 101–200, after averaging nRMSE over test initial states. Units below are 10^-3 per prediction step. This is a slope of the normalized error itself, not of log error.

| Task / method | Median | Mean | Sample SD | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kdv1d / isfno_h1_mse | 0.5751 | 0.5298 | 0.1857 | 0.7353 | 0.5751 | 0.2887 | 0.3919 | 0.6582 |
| kdv1d / sgno_zero_head_increment_v1 | 0.3176 | 2.711 | 5.38 | 0.2418 | 0.3176 | 12.33 | 0.2833 | 0.3778 |
| ks1d / isfno_h1_mse | 7.06 | 7.146 | 0.595 | 6.549 | 7.06 | 7.671 | 7.844 | 6.608 |
| ks1d / sgno_zero_head_increment_v1 | 6.006 | 6.028 | 0.202 | 5.988 | 6.3 | 5.743 | 6.105 | 6.006 |
| burgers1d_nu12e4 / isfno_h1_mse | 0.1039 | 0.1082 | 0.01424 | 0.1039 | 0.1066 | 0.09876 | 0.09875 | 0.1329 |
| burgers1d_nu12e4 / sgno_zero_head_increment_v1 | 0.06147 | 0.06313 | 0.01488 | 0.06147 | 0.06596 | 0.05143 | 0.04993 | 0.08688 |
| kolm2d / isfno_h1_mse | NF | NF | NF | NF | 0.8047 | 0.7964 | 1.455 | 0.5666 |
| kolm2d / sgno_zero_head_increment_v1 | 1.313 | 1.328 | 0.336 | 1.249 | 1.375 | 1.822 | 1.313 | 0.8824 |
