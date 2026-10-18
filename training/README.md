# Forecast model training: CVAE variants and XGBoost

For the two active low-level **XGBoost wind and RT-price forecasters**, see
[`xgboost/README.md`](xgboost/README.md). From the repository root:

```bash
python train_xgboost.py preflight
python train_xgboost.py wind
python train_xgboost.py rt --publication-lag 0
```

The existing T-CVAE instructions follow.

# Training the two Horse Creek T-CVAE variants

## Overview and model provenance

These are **different and checkpoint-incompatible** models. Both use the hourly
joint Horse Creek wind/DA-price historical development master packaged in this
repository. Training can run without the large raw historical training ZIP.

| | Thesis T-CVAE | Active continuous MPC CVAE v3 |
|---|---|---|
| Implementation | `training/thesis_tcvae/train_thesis_tcvae.py` | `forecast_release/source_releases/horse_creek_forecasting_release/train_continuous_cvae.py` |
| History / prediction | 96h / 96h | 96h / 120h (gap 0) |
| Architecture | CNN–LSTM; 4D context and latent; time-weighted MSE + KL | Conditional probabilistic CVAE; 16D latent; NLL + KL |
| Training years | 2018–2021 | 2018–2021, 2024 |
| Validation | 2024 | January–February 2025 |
| Reserved checks | 2025 development examples | 2025 Mar–Jun calibration; Jul–Dec development |
| Produces | `best_checkpoint.pt`, `last_checkpoint.pt` | `continuous_cvae_v3_checkpoint.pt` plus residual calibration `.npz` |

The thesis-style trainer is derived from the supplied corrected 17 October
2026 Pandas-3-compatible training package; its model, losses, splits and
hyperparameters are preserved. The source-input wrapper additionally reads
the bundled canonical CSV directly. The original runtime source file is
kept unchanged in `forecast_release/` because replay launchers include its
checksum in experiment provenance.

The active continuous trainer and calibrator are the original V3 scripts,
unaltered. **Retraining does not reproduce the existing frozen model weights
bit-for-bit unless environment and stochastic behavior also match.**

## Dependencies and included data

From repository root:

```bash
python -m pip install -r requirements-highs.txt
```

By default, both trainers read:

`forecast_release/source_releases/horse_creek_forecasting_release/data/development_master_hourly_2018_2021_2024_2025.csv`

This file already contains 2018–2021, part of 2024 and 2025. The
training commands do not use `holdout_inputs/` or any 2026 data.
For thesis training you may instead pass
`--input "C:\path\horse_creek_training_bundle_extended_2024_2025_QA.zip"`.
The ZIP reader prefers a `development_master_hourly` file if present,
otherwise selects eligible `TCVAE_hourly_wind_DA` development files.
The full raw download archive is not needed for either supplied trainer.

## A. Thesis T-CVAE, 96h/96h

Run an installation smoke test first:

```bash
python train_forecasts.py thesis --smoke --output training_runs/thesis_smoke
```

**Do not use smoke-test checkpoints for forecasts.** Full run:

```bash
python train_forecasts.py thesis --epochs 250 --batch 32
python train_forecasts.py check --day 2025-07-14 --dates 48 --scenarios 64
```

Outputs in `training_runs/thesis_tcvae/`:
`best_checkpoint.pt`, `last_checkpoint.pt`, `training_history.csv`,
`run_config.json`, and `conditioning_check/` with illustrative joint scenario
plots, per-day CRPS/coverage/MAE and conditioning diagnostics. Use an
independent 2025 check, not 2026 deployment data, for model selection.

On Windows, open `RUN_TCVAE_TRAINING_FROM_SPYDER.py` in Spyder and press F5.
Set `SMOKE_TEST_ONLY=True` for a quick installation check, then `False`.

**Scientific qualification:** The thesis-style model predicts from the start
of delivery day, given only history available by D-1 09:30. There is an
intentional time gap between the last observed hour and the first delivery
hour. It is *not* the current MPC continuous from-issue-time scenario model.
The diagnostics include a post-issue visualization interval, but the
intervening realized values must never be used as model inputs.

## B. Continuous calibrated CVAE used in the MPC

Run an installation smoke test:

```bash
python train_forecasts.py continuous --smoke --output training_runs/continuous_smoke
```

Then train with the original calibrated-CVAE method and fit the residual
bootstrap on pre-July 2025 observations:

```bash
python train_forecasts.py continuous --epochs 80
python train_forecasts.py calibrate
```

Outputs in `training_runs/continuous_cvae/`:
`continuous_cvae_v3_checkpoint.pt`, `model_manifest.json`,
`training_history.csv`, `continuous_calibration_residual_paths.npz`,
`calibration_manifest.json`.

The split is **2018–2021 and 2024 training / 2025 Jan–Feb early stopping /
2025 Mar–Jun calibration / 2025 Jul–Dec development test**. The calibration
script has its own pre-July cutoff and uses paired wind–price residual paths.
This workflow trains a compatible active-MPC architecture, but installing a
new checkpoint requires explicit evaluation, fresh inference and replay;
**no deployment command is included** and the frozen files under
`forecast_release/.../models/` are never modified.

On Windows, use `RUN_CONTINUOUS_CVAE_TRAINING_FROM_SPYDER.py` (F5).

## Research safeguards

- The data path is recorded in each thesis `run_config.json` together with its SHA256.
- Thesis training windows cannot straddle the chronological splits; checks
  require 96 complete historical hours and 96 complete future hours.
- The corrected timestamp handling explicitly casts to nanoseconds (pandas 3 compatible).
- Both models write only into gitignored `training_runs/`, separate from active MPC artifacts.
- 2026 primary holdouts remain evaluation-only; don't use those observations for model tuning.
- Training checks are smoke tests, not independent probabilistic forecast validation.
