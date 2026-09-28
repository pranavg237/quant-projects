# Fama-French 5-factor analysis

Factors: Kenneth R. French Data Library (monthly). Asset returns: Yahoo Finance (split- and dividend-adjusted closes). Standard errors: Newey-West (HAC). Alphas and returns are annualized.

## Fama-French 5-factor regressions

|  | alpha (ann.) | t(alpha) | Mkt-RF | SMB | HML | RMW | CMA | R2 | adj R2 | resid vol (ann.) | IR | N |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | -0.36% | -1.99 | 0.99 | -0.12 | 0.02 | 0.05 | 0.02 | 0.997 | 0.996 | 0.88% | -0.41 | 259 |
| IWN | -1.33% | -2.22 | 0.96 | 0.82 | 0.36 | 0.02 | 0.01 | 0.982 | 0.982 | 2.70% | -0.49 | 259 |
| BRK-B | 2.76% | 1.03 | 0.68 | -0.36 | 0.43 | 0.08 | 0.10 | 0.396 | 0.384 | 13.63% | 0.20 | 259 |
| AAPL | 15.83% | 2.95 | 1.24 | -0.12 | -0.44 | 0.62 | -0.43 | 0.402 | 0.390 | 23.97% | 0.66 | 259 |
| Portfolio | -0.75% | -2.58 | 0.98 | 0.25 | 0.16 | 0.04 | 0.02 | 0.994 | 0.994 | 1.30% | -0.58 | 259 |

<sub>Data: [summary.csv](summary.csv)</sub>

## SPY

Jan 2005 to Jul 2026, 259 observations. Alpha -0.36% a year (t = -1.99), R² 0.997, adjusted R² 0.996, residual volatility 0.88%, information ratio -0.41.

|  | coef | std err | t | p-value | ci low (95%) | ci high (95%) |
|:---|---:|---:|---:|---:|---:|---:|
| alpha | -0.0003 | 0.0002 | -1.9865 | 0.0470 | -0.0006 | -0.0000 |
| Mkt-RF | 0.9947 | 0.0056 | 178.6882 | 0.0000 | 0.9838 | 1.0056 |
| SMB | -0.1236 | 0.0072 | -17.0677 | 0.0000 | -0.1378 | -0.1094 |
| HML | 0.0244 | 0.0089 | 2.7315 | 0.0063 | 0.0069 | 0.0419 |
| RMW | 0.0451 | 0.0137 | 3.3027 | 0.0010 | 0.0183 | 0.0719 |
| CMA | 0.0231 | 0.0111 | 2.0776 | 0.0377 | 0.0013 | 0.0449 |

<sub>Data: [spy-coefficients.csv](spy-coefficients.csv)</sub>

![SPY factor loadings](spy-loadings.png)

### SPY: return attribution

|  | return (ann.) | share of return | share of variance |
|:---|---:|---:|---:|
| alpha | -0.36% | -3.74% | 0.00% |
| Mkt-RF | 9.88% | 101.90% | 102.11% |
| SMB | 0.03% | 0.36% | -2.15% |
| HML | -0.00% | -0.04% | 0.20% |
| RMW | 0.14% | 1.46% | -0.36% |
| CMA | 0.01% | 0.07% | -0.14% |
| residual | -0.00% | -0.00% | 0.34% |
| total | 9.70% | 100.00% | 100.00% |

<sub>Data: [spy-attribution.csv](spy-attribution.csv)</sub>

![SPY return attribution](spy-attribution-chart.png)

![SPY actual vs factor-explained return](spy-cumulative.png)

<sub>Chart data: [spy-cumulative.csv](spy-cumulative.csv)</sub>

### SPY: model comparison (common sample)

|  | alpha (ann.) | t(alpha) | Mkt-RF | SMB | HML | RMW | CMA | MOM | R2 | adj R2 | resid vol (ann.) | IR | N | BIC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | 0.12% | 0.37 | 0.96 |  |  |  |  |  | 0.990 | 0.990 | 1.46% | 0.08 | 259 | -2088.98 |
| Fama-French 3-factor | -0.20% | -1.04 | 0.99 | -0.14 | 0.01 |  |  |  | 0.996 | 0.996 | 0.93% | -0.21 | 259 | -2315.98 |
| Carhart 4-factor | -0.18% | -0.95 | 0.99 | -0.14 | 0.01 |  |  | -0.00 | 0.996 | 0.996 | 0.93% | -0.20 | 259 | -2311.15 |
| Fama-French 5-factor | -0.36% | -1.99 | 0.99 | -0.12 | 0.02 | 0.05 | 0.02 |  | 0.997 | 0.996 | 0.88% | -0.41 | 259 | -2332.51 |
| Fama-French 6-factor | -0.35% | -1.92 | 0.99 | -0.12 | 0.02 | 0.04 | 0.02 | -0.00 | 0.997 | 0.996 | 0.88% | -0.39 | 259 | -2327.71 |

<sub>Data: [spy-models.csv](spy-models.csv)</sub>

### SPY: rolling 36-period estimates

|  | alpha (ann.) | Mkt-RF | SMB | HML | RMW | CMA | R2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| latest | -0.30% | 1.00 | -0.08 | 0.03 | 0.03 | 0.01 | 0.997 |
| mean | -0.28% | 0.99 | -0.12 | 0.01 | 0.06 | 0.03 | 0.997 |
| min | -1.58% | 0.95 | -0.16 | -0.04 | -0.02 | -0.06 | 0.985 |
| max | 0.41% | 1.04 | -0.07 | 0.06 | 0.12 | 0.07 | 0.999 |

<sub>Data: [spy-rolling-summary.csv](spy-rolling-summary.csv)</sub>

![SPY rolling estimates](spy-rolling.png)

<sub>Chart data: [spy-rolling.csv](spy-rolling.csv)</sub>

## IWN

Jan 2005 to Jul 2026, 259 observations. Alpha -1.33% a year (t = -2.22), R² 0.982, adjusted R² 0.982, residual volatility 2.70%, information ratio -0.49.

|  | coef | std err | t | p-value | ci low (95%) | ci high (95%) |
|:---|---:|---:|---:|---:|---:|---:|
| alpha | -0.0011 | 0.0005 | -2.2153 | 0.0267 | -0.0021 | -0.0001 |
| Mkt-RF | 0.9611 | 0.0134 | 71.6913 | 0.0000 | 0.9349 | 0.9874 |
| SMB | 0.8213 | 0.0246 | 33.3470 | 0.0000 | 0.7730 | 0.8696 |
| HML | 0.3618 | 0.0251 | 14.4261 | 0.0000 | 0.3126 | 0.4109 |
| RMW | 0.0243 | 0.0262 | 0.9283 | 0.3532 | -0.0270 | 0.0756 |
| CMA | 0.0113 | 0.0344 | 0.3281 | 0.7429 | -0.0562 | 0.0788 |

<sub>Data: [iwn-coefficients.csv](iwn-coefficients.csv)</sub>

![IWN factor loadings](iwn-loadings.png)

### IWN: return attribution

|  | return (ann.) | share of return | share of variance |
|:---|---:|---:|---:|
| alpha | -1.33% | -16.64% | 0.00% |
| Mkt-RF | 9.55% | 119.29% | 64.48% |
| SMB | -0.23% | -2.90% | 25.97% |
| HML | -0.06% | -0.74% | 7.99% |
| RMW | 0.08% | 0.95% | -0.21% |
| CMA | 0.00% | 0.04% | 0.01% |
| residual | -0.00% | -0.00% | 1.76% |
| total | 8.00% | 100.00% | 100.00% |

<sub>Data: [iwn-attribution.csv](iwn-attribution.csv)</sub>

![IWN return attribution](iwn-attribution-chart.png)

![IWN actual vs factor-explained return](iwn-cumulative.png)

<sub>Chart data: [iwn-cumulative.csv](iwn-cumulative.csv)</sub>

### IWN: model comparison (common sample)

|  | alpha (ann.) | t(alpha) | Mkt-RF | SMB | HML | RMW | CMA | MOM | R2 | adj R2 | resid vol (ann.) | IR | N | BIC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | -3.49% | -1.54 | 1.16 |  |  |  |  |  | 0.777 | 0.776 | 9.53% | -0.37 | 259 | -1117.41 |
| Fama-French 3-factor | -1.10% | -1.70 | 0.97 | 0.82 | 0.52 |  |  |  | 0.980 | 0.980 | 2.84% | -0.39 | 259 | -1736.06 |
| Carhart 4-factor | -1.11% | -1.67 | 0.97 | 0.82 | 0.52 |  |  | 0.00 | 0.980 | 0.980 | 2.84% | -0.39 | 259 | -1730.59 |
| Fama-French 5-factor | -1.33% | -2.22 | 0.96 | 0.82 | 0.36 | 0.02 | 0.01 |  | 0.982 | 0.982 | 2.70% | -0.49 | 259 | -1752.81 |
| Fama-French 6-factor | -1.39% | -2.22 | 0.96 | 0.82 | 0.37 | 0.03 | 0.01 | 0.01 | 0.982 | 0.982 | 2.70% | -0.51 | 259 | -1748.19 |

<sub>Data: [iwn-models.csv](iwn-models.csv)</sub>

### IWN: rolling 36-period estimates

|  | alpha (ann.) | Mkt-RF | SMB | HML | RMW | CMA | R2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| latest | -0.62% | 0.99 | 0.84 | 0.44 | -0.00 | 0.04 | 0.985 |
| mean | -1.46% | 0.97 | 0.81 | 0.36 | 0.07 | 0.04 | 0.984 |
| min | -4.88% | 0.91 | 0.68 | 0.15 | -0.12 | -0.16 | 0.956 |
| max | 1.83% | 1.08 | 0.96 | 0.60 | 0.27 | 0.25 | 0.994 |

<sub>Data: [iwn-rolling-summary.csv](iwn-rolling-summary.csv)</sub>

![IWN rolling estimates](iwn-rolling.png)

<sub>Chart data: [iwn-rolling.csv](iwn-rolling.csv)</sub>

## BRK-B

Jan 2005 to Jul 2026, 259 observations. Alpha +2.76% a year (t = 1.03), R² 0.396, adjusted R² 0.384, residual volatility 13.63%, information ratio 0.20.

|  | coef | std err | t | p-value | ci low (95%) | ci high (95%) |
|:---|---:|---:|---:|---:|---:|---:|
| alpha | 0.0023 | 0.0022 | 1.0285 | 0.3037 | -0.0021 | 0.0067 |
| Mkt-RF | 0.6812 | 0.0823 | 8.2730 | 0.0000 | 0.5198 | 0.8426 |
| SMB | -0.3614 | 0.0988 | -3.6576 | 0.0003 | -0.5551 | -0.1677 |
| HML | 0.4310 | 0.1189 | 3.6255 | 0.0003 | 0.1980 | 0.6640 |
| RMW | 0.0849 | 0.1083 | 0.7843 | 0.4329 | -0.1273 | 0.2971 |
| CMA | 0.0967 | 0.1737 | 0.5569 | 0.5776 | -0.2437 | 0.4372 |

<sub>Data: [brk-b-coefficients.csv](brk-b-coefficients.csv)</sub>

![BRK-B factor loadings](brk-b-loadings.png)

### BRK-B: return attribution

|  | return (ann.) | share of return | share of variance |
|:---|---:|---:|---:|
| alpha | 2.76% | 27.99% | 0.00% |
| Mkt-RF | 6.77% | 68.71% | 33.05% |
| SMB | 0.10% | 1.04% | -1.93% |
| HML | -0.07% | -0.71% | 8.18% |
| RMW | 0.27% | 2.71% | -0.04% |
| CMA | 0.03% | 0.27% | 0.38% |
| residual | -0.00% | -0.00% | 60.36% |
| total | 9.85% | 100.00% | 100.00% |

<sub>Data: [brk-b-attribution.csv](brk-b-attribution.csv)</sub>

![BRK-B return attribution](brk-b-attribution-chart.png)

![BRK-B actual vs factor-explained return](brk-b-cumulative.png)

<sub>Chart data: [brk-b-cumulative.csv](brk-b-cumulative.csv)</sub>

### BRK-B: model comparison (common sample)

|  | alpha (ann.) | t(alpha) | Mkt-RF | SMB | HML | RMW | CMA | MOM | R2 | adj R2 | resid vol (ann.) | IR | N | BIC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | 3.65% | 1.21 | 0.62 |  |  |  |  |  | 0.303 | 0.300 | 14.54% | 0.25 | 259 | -898.40 |
| Fama-French 3-factor | 3.01% | 1.17 | 0.67 | -0.44 | 0.41 |  |  |  | 0.402 | 0.395 | 13.51% | 0.22 | 259 | -927.23 |
| Carhart 4-factor | 3.21% | 1.22 | 0.66 | -0.44 | 0.39 |  |  | -0.05 | 0.404 | 0.394 | 13.52% | 0.24 | 259 | -922.25 |
| Fama-French 5-factor | 2.76% | 1.03 | 0.68 | -0.36 | 0.43 | 0.08 | 0.10 |  | 0.396 | 0.384 | 13.63% | 0.20 | 259 | -913.59 |
| Fama-French 6-factor | 2.99% | 1.08 | 0.67 | -0.37 | 0.41 | 0.07 | 0.11 | -0.05 | 0.398 | 0.383 | 13.64% | 0.22 | 259 | -908.64 |

<sub>Data: [brk-b-models.csv](brk-b-models.csv)</sub>

### BRK-B: rolling 36-period estimates

|  | alpha (ann.) | Mkt-RF | SMB | HML | RMW | CMA | R2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| latest | -5.65% | 0.48 | -0.39 | 1.04 | 0.13 | -0.07 | 0.232 |
| mean | 5.57% | 0.69 | -0.44 | 0.33 | -0.35 | 0.15 | 0.554 |
| min | -9.63% | -0.13 | -1.05 | -1.20 | -2.69 | -0.83 | 0.130 |
| max | 26.31% | 1.13 | 0.06 | 1.11 | 0.67 | 1.33 | 0.821 |

<sub>Data: [brk-b-rolling-summary.csv](brk-b-rolling-summary.csv)</sub>

![BRK-B rolling estimates](brk-b-rolling.png)

<sub>Chart data: [brk-b-rolling.csv](brk-b-rolling.csv)</sub>

## AAPL

Jan 2005 to Jul 2026, 259 observations. Alpha +15.83% a year (t = 2.95), R² 0.402, adjusted R² 0.390, residual volatility 23.97%, information ratio 0.66.

|  | coef | std err | t | p-value | ci low (95%) | ci high (95%) |
|:---|---:|---:|---:|---:|---:|---:|
| alpha | 0.0132 | 0.0045 | 2.9538 | 0.0031 | 0.0044 | 0.0219 |
| Mkt-RF | 1.2371 | 0.1062 | 11.6533 | 0.0000 | 1.0290 | 1.4452 |
| SMB | -0.1189 | 0.1620 | -0.7337 | 0.4631 | -0.4365 | 0.1987 |
| HML | -0.4399 | 0.1896 | -2.3197 | 0.0204 | -0.8116 | -0.0682 |
| RMW | 0.6219 | 0.2021 | 3.0773 | 0.0021 | 0.2258 | 1.0181 |
| CMA | -0.4347 | 0.2975 | -1.4614 | 0.1439 | -1.0177 | 0.1483 |

<sub>Data: [aapl-coefficients.csv](aapl-coefficients.csv)</sub>

![AAPL factor loadings](aapl-loadings.png)

### AAPL: return attribution

|  | return (ann.) | share of return | share of variance |
|:---|---:|---:|---:|
| alpha | 15.83% | 52.67% | 0.00% |
| Mkt-RF | 12.29% | 40.88% | 35.36% |
| SMB | 0.03% | 0.11% | -0.27% |
| HML | 0.07% | 0.24% | 2.45% |
| RMW | 1.95% | 6.50% | 0.15% |
| CMA | -0.12% | -0.40% | 2.51% |
| residual | -0.00% | -0.00% | 59.80% |
| total | 30.06% | 100.00% | 100.00% |

<sub>Data: [aapl-attribution.csv](aapl-attribution.csv)</sub>

![AAPL return attribution](aapl-attribution-chart.png)

![AAPL actual vs factor-explained return](aapl-cumulative.png)

<sub>Chart data: [aapl-cumulative.csv](aapl-cumulative.csv)</sub>

### AAPL: model comparison (common sample)

|  | alpha (ann.) | t(alpha) | Mkt-RF | SMB | HML | RMW | CMA | MOM | R2 | adj R2 | resid vol (ann.) | IR | N | BIC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | 18.66% | 3.28 | 1.15 |  |  |  |  |  | 0.328 | 0.326 | 25.21% | 0.74 | 259 | -613.22 |
| Fama-French 3-factor | 17.50% | 3.33 | 1.24 | -0.26 | -0.59 |  |  |  | 0.380 | 0.373 | 24.31% | 0.72 | 259 | -622.92 |
| Carhart 4-factor | 17.70% | 3.33 | 1.23 | -0.27 | -0.61 |  |  | -0.05 | 0.380 | 0.371 | 24.35% | 0.73 | 259 | -617.54 |
| Fama-French 5-factor | 15.83% | 2.95 | 1.24 | -0.12 | -0.44 | 0.62 | -0.43 |  | 0.402 | 0.390 | 23.97% | 0.66 | 259 | -621.14 |
| Fama-French 6-factor | 15.79% | 2.89 | 1.24 | -0.12 | -0.44 | 0.62 | -0.44 | 0.01 | 0.402 | 0.388 | 24.02% | 0.66 | 259 | -615.59 |

<sub>Data: [aapl-models.csv](aapl-models.csv)</sub>

### AAPL: rolling 36-period estimates

|  | alpha (ann.) | Mkt-RF | SMB | HML | RMW | CMA | R2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| latest | 1.25% | 0.85 | -0.01 | 0.00 | 0.19 | -0.29 | 0.263 |
| mean | 14.10% | 1.29 | -0.12 | -0.36 | 0.86 | -1.11 | 0.552 |
| min | -9.22% | 0.45 | -1.69 | -1.81 | -0.85 | -3.56 | 0.243 |
| max | 49.35% | 3.01 | 0.53 | 1.03 | 2.96 | 1.24 | 0.813 |

<sub>Data: [aapl-rolling-summary.csv](aapl-rolling-summary.csv)</sub>

![AAPL rolling estimates](aapl-rolling.png)

<sub>Chart data: [aapl-rolling.csv](aapl-rolling.csv)</sub>

## Portfolio

Jan 2005 to Jul 2026, 259 observations. Alpha -0.75% a year (t = -2.58), R² 0.994, adjusted R² 0.994, residual volatility 1.30%, information ratio -0.58.

|  | coef | std err | t | p-value | ci low (95%) | ci high (95%) |
|:---|---:|---:|---:|---:|---:|---:|
| alpha | -0.0006 | 0.0002 | -2.5808 | 0.0099 | -0.0011 | -0.0002 |
| Mkt-RF | 0.9813 | 0.0069 | 142.7619 | 0.0000 | 0.9678 | 0.9947 |
| SMB | 0.2544 | 0.0125 | 20.3692 | 0.0000 | 0.2299 | 0.2789 |
| HML | 0.1594 | 0.0126 | 12.6933 | 0.0000 | 0.1348 | 0.1840 |
| RMW | 0.0368 | 0.0163 | 2.2558 | 0.0241 | 0.0048 | 0.0687 |
| CMA | 0.0184 | 0.0168 | 1.0948 | 0.2736 | -0.0145 | 0.0513 |

<sub>Data: [portfolio-coefficients.csv](portfolio-coefficients.csv)</sub>

![Portfolio factor loadings](portfolio-loadings.png)

### Portfolio: return attribution

|  | return (ann.) | share of return | share of variance |
|:---|---:|---:|---:|
| alpha | -0.75% | -8.32% | 0.00% |
| Mkt-RF | 9.75% | 108.07% | 89.91% |
| SMB | -0.07% | -0.80% | 7.08% |
| HML | -0.03% | -0.29% | 2.78% |
| RMW | 0.12% | 1.28% | -0.34% |
| CMA | 0.01% | 0.06% | -0.04% |
| residual | -0.00% | -0.00% | 0.62% |
| total | 9.02% | 100.00% | 100.00% |

<sub>Data: [portfolio-attribution.csv](portfolio-attribution.csv)</sub>

![Portfolio return attribution](portfolio-attribution-chart.png)

![Portfolio actual vs factor-explained return](portfolio-cumulative.png)

<sub>Chart data: [portfolio-cumulative.csv](portfolio-cumulative.csv)</sub>

### Portfolio: model comparison (common sample)

|  | alpha (ann.) | t(alpha) | Mkt-RF | SMB | HML | RMW | CMA | MOM | R2 | adj R2 | resid vol (ann.) | IR | N | BIC |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAPM | -1.33% | -1.52 | 1.04 |  |  |  |  |  | 0.955 | 0.955 | 3.48% | -0.38 | 259 | -1638.56 |
| Fama-French 3-factor | -0.56% | -1.80 | 0.98 | 0.24 | 0.22 |  |  |  | 0.993 | 0.993 | 1.37% | -0.41 | 259 | -2112.68 |
| Carhart 4-factor | -0.55% | -1.77 | 0.98 | 0.24 | 0.22 |  |  | -0.00 | 0.993 | 0.993 | 1.37% | -0.40 | 259 | -2107.14 |
| Fama-French 5-factor | -0.75% | -2.58 | 0.98 | 0.25 | 0.16 | 0.04 | 0.02 |  | 0.994 | 0.994 | 1.30% | -0.58 | 259 | -2131.09 |
| Fama-French 6-factor | -0.76% | -2.61 | 0.98 | 0.25 | 0.16 | 0.04 | 0.02 | 0.00 | 0.994 | 0.994 | 1.30% | -0.59 | 259 | -2125.74 |

<sub>Data: [portfolio-models.csv](portfolio-models.csv)</sub>

### Portfolio: rolling 36-period estimates

|  | alpha (ann.) | Mkt-RF | SMB | HML | RMW | CMA | R2 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| latest | -0.43% | 0.99 | 0.28 | 0.19 | 0.02 | 0.02 | 0.994 |
| mean | -0.75% | 0.98 | 0.25 | 0.15 | 0.06 | 0.03 | 0.994 |
| min | -2.18% | 0.95 | 0.18 | 0.06 | -0.05 | -0.05 | 0.982 |
| max | 0.54% | 1.05 | 0.32 | 0.25 | 0.14 | 0.12 | 0.998 |

<sub>Data: [portfolio-rolling-summary.csv](portfolio-rolling-summary.csv)</sub>

![Portfolio rolling estimates](portfolio-rolling.png)

<sub>Chart data: [portfolio-rolling.csv](portfolio-rolling.csv)</sub>
