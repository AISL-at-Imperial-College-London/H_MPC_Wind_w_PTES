# Horse Creek: matched 96h-history / 96h-forecast benchmark

## What this experiment compares

- **Frozen continuous CVAE used in the MPC paper**: 96 hourly wind+DA history, originally outputs 120 forecast hours; score its first 96 forecast hours, without altering checkpoints, weights or residual calibration.
- **New empirical-copula / historical-analogue predictor**: 96 hourly wind+DA history for joint analogue matching and DA-price quantile recalibration; 96 future hours of paired wind+price observations sampled from a seasonally filtered, frozen, pre-July-2025 history bank.
- **Identical**: issue time D-1 09:30 America/Chicago; latest completed hourly measurements; 96 forecast target timestamps starting at the first unobserved hour; realized 2026 test data; raw sample size (500), reduction rule (same backward deletion), reduced sample size (50), evaluation scores and random seed identifiers.

This is **not** an exact recreation of the 84-hour delivery-aligned MPC forecast and **not** a replacement of the paper's frozen CVAE checkpoint. The 96-hour experiment uses a new empirical-analogue configuration; both its 96-hour wind features and 96-hour price window differ from the original copula benchmark. The original deployed CVAE actually generates 120 hours; only the first 96 are scored.

## Data and historical library

The comparison runs against the existing Horse Creek MPC repository's `forecast_release/source_releases/horse_creek_forecasting_release` checkpoint, calibration, historical data and `holdout_inputs/{winter,spring,summer}` folders. No additional data downloads are necessary. The copula reference bank uses only complete 96h-history + 96h-future paired trajectories from history **through 30 June 2025**; candidate forecasts are at the same local hour as the current issue and within ±2 forecast-issue calendar months. This freezes the copula reference before all 2026 holdouts and near the end of the CVAE's 2025 residual calibration window. It is not identical training data to the CVAE: the model architectures and their historical use of data remain different.

The analogue weights use a Gaussian of a fixed standardised feature distance plus 0.15 uniform weight. Features (derived from each 96-hour history) are the wind and transformed-DA-price means in each consecutive 24-hour block, last-24-hour standard deviations, and last-12-hour means; the four blocks have fixed recency weights. The historical sampled DA-price ranks are monotone-mapped onto the empirical distribution of the **last 96 completed DA-price observations**. Both sources are read causally before the issue cutoff.

## Run

Extract the original `Horse_Creek_PTES_MPC_Repo_With_Copula_CVAE_Benchmark.zip` or later equivalent, then run from the directory containing `benchmark_96x96.py`:

```bash
python benchmark_96x96.py --root /path/to/horse-creek-ptes-mpc --seeds 3714 13714 2718 --out results_96x96
```

Environment: Python with `numpy`, `pandas`, `scipy`, `torch` and the packages listed in the repository's open-source requirements. **Gurobi is not needed** for this forecast-only comparison.

The package contains `daily_metrics.csv` (every issue/seed), `weekly_metrics.csv`, `pooled_metrics.csv`, `multi_seed_metrics.csv`, `issuance_audit.csv` and `methodology.json`. Scores include exact empirical weighted CRPS, weighted nominal 90% coverage, ensemble-mean MAE, and a 96-hour joint energy score with wind scaled by 230 MW and DA-price mapped using `asinh(price/50)`; the energy score is dimensionless.

**Do not combine this table with the previous 84-hour horizon table** without explicitly labeling the different forecast scoring windows. Results do not establish superior financial returns, and overlapping 96-hour origins are not independent statistical replicates.

## Scope and limitations

- Four daily issue origins per unseen week (12 across February, May and July 2026); 3 Monte Carlo seeds. Test weeks have limited archived post-week target data.
- This is a single specified analogue-distance kernel, not an exhaustive tuning study; price empirical calibration with only 96 samples can underestimate tail uncertainty.
- CVAE sampled paths are still influenced by its own historical multi-year training and 2025 residual-calibration bank; matching the *recent information set* does not mean matching the *training algorithms*.
- No closed-loop MPC optimisation, settlement, or comparative revenue experiment was run for the new copula scenarios.
