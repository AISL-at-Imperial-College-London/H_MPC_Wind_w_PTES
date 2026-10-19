# Comparison of the historical copula and calibrated continuous CVAE

## Executive finding

Using the **unmodified forecast providers shipped in the Horse Creek MPC repository**, the comparison does not demonstrate a robust overall forecasting advantage for the continuous CVAE over the causal historical copula. On four post-calibration 2025 weeks (28 daily issuances), the 50-scenario copula has slightly lower wind CRPS; the calibrated CVAE has slightly lower DA-price CRPS and 84-hour energy score at the manuscript's original seed 3714. **The small aggregate DA-price and energy-score advantages reverse at an alternative scenario-generation seed (13714).** Model rankings also reverse across weeks. The comparison is forecast-only, not a storage-revenue comparison.

The **February 2026 unseen holdout** was additionally evaluated where the archive supports causally valid input histories and realised targets. The wind/price split remains: copula lower wind CRPS, CVAE lower DA-price CRPS. Only four Feb dates support complete 84-hour truth, and six support complete 24-hour truth. May and July 2026 cannot be evaluated with the original copula without acquiring additional pre-issuance DA-price history.

## Experiment protocol

- Frozen forecasters: `horse_creek_forecasts.JointForecastProvider(method='tcvae')` (the calibrated continuous 120-hour CVAE) versus `method='copula'` (paired historical-analogue wind/DA empirical copula with price-rank recalibration). **Not** the separate 250-epoch thesis 96/96 T-CVAE.
- CVAE training: 2018–2021 and 2024; early-stopping validation Jan–Feb 2025; calibration residuals Mar–Jun 2025. The copula uses historical blocks completed before each issue date (2018–2025 source data) and recent eligible DA prices. These are not identical learning/calibration designs; conclusions relate to these implemented forecasting systems.
- Daily issuance at D−1 09:30 **America/Chicago**, evaluated from 00:00 local on delivery day D. Each forecaster produces 500 paired scenarios, reduced with identical weighted original backward-deletion procedure to 50 representatives over the 84-hour MPC slice. Seed 3714 unless otherwise stated.
- Four 7-day 2025 periods: July 14–20; September 8–14; November 10–16; December 8–14. 84 future hourly values scored per issue. Model inputs strictly precede the issue cutoff; realisations are accessed only for scoring. **Daily 84-hour horizons overlap**, so 2,352 forecast-target comparisons should not be treated as independent observations.
- Supplementary February 2026 holdout: 84-hour scoring on Feb 9–12 (4 issues); 24-hour scoring on Feb 9–14 (6 issues). 2026 historical wind/DA observations passed to the providers are explicitly truncated at the issue time. For winter, the copula obtains its ≥45 days of recent DA context from late 2025 plus the available Jan 29–Feb 2026 records; Jan 1–28 2026 are absent from the archive.
- Full raw 500 and post-reduction 50 forecasts are scored separately. Original provider code, model weights, calibration, scenario-selection algorithm and forecast-input data were not changed.

## Overall 2025 results, 50 weighted scenarios

| Metric | Copula | Continuous CVAE | Favours |
|---|---:|---:|---|
| Wind CRPS (MW) ↓ | **39.341** | 40.218 | Copula |
| DA-price CRPS (USD/MWh) ↓ | 9.278 | **9.133** | CVAE |
| Wind ensemble-mean MAE (MW) ↓ | **57.492** | 58.241 | Copula |
| DA-price ensemble-mean MAE (USD/MWh) ↓ | 13.437 | **12.710** | CVAE |
| Wind 90% central interval coverage (target 90%) | 91.7% | **90.2%** | CVAE closer to target |
| DA-price 90% central interval coverage (target 90%) | 80.0% | **88.1%** | CVAE closer to target |
| Joint full-trajectory energy score ↓ | 2.640 | **2.631** | Very small CVAE difference |

CRPS is a proper probabilistic score combining sharpness and calibration (lower is better). Energy score is computed over the complete paired temporal path, using wind/230 and asinh(price/50) coordinates. Its numerical units are dimensionless and not directly comparable to CRPS.

### Effect of scenario reduction

| Raw-500 metric | Copula | CVAE |
|---|---:|---:|
| Wind CRPS (MW) | **38.746** | 39.969 |
| DA-price CRPS (USD/MWh) | **8.939** | 9.074 |
| Joint energy score | **2.570** | 2.607 |

With raw samples, the copula is slightly better on all three scores. The price and energy ranking changes at 50 reduced scenarios for seed 3714. Thus the decision-facing distribution and the full generative distribution should be evaluated separately.

## Results by 2025 week (50 weighted scenarios)

| Delivery-start week | Wind CRPS Copula | Wind CRPS CVAE | DA-price CRPS Copula | DA-price CRPS CVAE | Joint energy Copula | Joint energy CVAE |
|---|---:|---:|---:|---:|---:|---:|
| July 14–20 | 43.990 | **41.973** | 8.231 | **5.994** | 2.652 | **2.259** |
| September 8–14 | **28.285** | 31.301 | 7.747 | **5.701** | 2.121 | **1.886** |
| November 10–16 | **41.502** | 44.325 | **11.460** | 13.317 | **2.981** | 3.325 |
| December 8–14 | 43.588 | **43.271** | **9.672** | 11.520 | **2.808** | 3.053 |

## Winter 2026 unseen holdout, 50 weighted scenarios

| Metric | Copula | CVAE |
|---|---:|---:|
| 84h wind CRPS, Feb 9–12 | **41.697** | 43.681 |
| 84h DA-price CRPS, Feb 9–12 | 7.119 | **6.268** |
| 24h wind CRPS, Feb 9–14 | **43.311** | 46.757 |
| 24h DA-price CRPS, Feb 9–14 | 7.790 | **7.170** |
| 84h joint energy, Feb 9–12 | **2.498** | 2.504 |

The 84-hour and 24-hour comparisons have *different issue-date samples* and must not be directly compared as a lead-time improvement. This winter test is restricted by the archived end date of hourly outcomes. It does not establish the same ranking on May or July 2026.

## Random scenario draw sensitivity

An additional full 2025 run with seed **13714** changes the 50-scenario aggregate scores:

| Metric | Seed | Copula | CVAE |
|---|---|---:|---:|
| Wind CRPS | 3714 | **39.341** | 40.218 |
| Wind CRPS | 13714 | **38.910** | 40.582 |
| DA-price CRPS | 3714 | 9.278 | **9.133** |
| DA-price CRPS | 13714 | **9.133** | 9.233 |
| Joint energy score | 3714 | 2.640 | **2.631** |
| Joint energy score | 13714 | **2.597** | 2.641 |

Therefore the CVAE's small aggregate 50-scenario DA-price and energy score advantages should **not** be described as consistent or statistically established. The larger week-by-week differences in both directions, and wind's relative copula advantage, are more relevant than declaring an aggregate winner.

A descriptive 4-week block bootstrap for the CVAE-minus-copula difference at seed 3714 gives wide approximate 95% intervals: wind CRPS [−1.17,+2.92], DA-price CRPS [−2.14,+1.85], joint energy [−0.313,+0.295]. All include zero, and four weeks provide too few independent clusters for a strong statistical inference.

## Dependence and extremes

The scored CSVs also include realised/scenario wind–DA timewise Spearman correlation, Brier scores for wind below 10 MW, prices above 100 USD/MWh, and simultaneous low wind/high prices. These metrics help diagnose dependence but **do not on their own validate the entire joint density**, especially when extremes are rare in the selected weeks. Interpret them alongside the joint-path energy score and calibration.

## How to rerun in the public repository

From the repository root, with the documented dependencies installed:

```bash
python analysis/forecast_comparison/compare_copula_cvae.py
python analysis/forecast_comparison/compare_copula_cvae.py --weeks feb_2026_84h
python analysis/forecast_comparison/compare_copula_cvae.py --weeks feb_2026_24h
python analysis/forecast_comparison/compare_copula_cvae.py --seed 13714
```

Each invocation writes its output to `forecast_comparison_results/` by default; **use `--out SOME_NEW_FOLDER` for each run** to prevent result collisions. No Gurobi licence is required because this is a forecast-only benchmark.

The comparison script reads the versioned frozen checkpoints and the archived historical CSVs; it never overwrites them. Each run produces daily and hourly CRPS, interval, MAE and tail-event metrics, aggregate CSVs, detailed issuance metadata (including forecast as-of cutoff), and model/data SHA-256 hashes in `comparison_method.json`. Reported numeric values refer to the committed reference checkpoints and should not be assumed to hold after retraining or checkpoint replacement.

### Unresolved scope

- The May and July 2026 comparison cannot run with the *original* copula because the ZIPs do not contain the required 45 days of recent completed DA-price observations within 90 days of each forecast issuance. The original provider fails explicitly rather than silently imputing prices. Obtaining the missing market history is preferable to relaxing the method's as-of constraints.
- This study does not compare simulated revenues, DA positions, PTES dispatch, or solver performance. To quantify value to control, a subsequent stochastic HL+LL replay must be run for **both** forecast providers with identical optimisation settings, initial inventories and market settlements.
- Operational DA publication times for historical 2025 values have not been individually audited; both providers consume only fully completed historical hourly prices. A future fully event-sourced replay should separately verify real-time publication provenance.
- Scored 2025 weeks include periods available to the research team during development and should not be presented as the paper's untouched 2026 test-set results.
