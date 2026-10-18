# Low-level MPC XGBoost training — wind and RT15 settlement prices

The repository includes the original V3.3 training source for both active XGBoost
families, plus a safety wrapper with isolated output directories. The training
CSV files are already bundled. **Do not replace the frozen production model
files automatically**: these were used in the paper evaluation and must remain
unaltered for reproducibility.

## 1. Run

From the repository root:

```bash
python -m pip install -r requirements-highs.txt
python train_xgboost.py preflight
python train_xgboost.py wind
python train_xgboost.py rt --publication-lag 0
# Or both sequentially:
python train_xgboost.py all
```

Windows/Spyder: run `RUN_XGBOOST_TRAINING_FROM_SPYDER.py` using F5.
`MODEL` may be `wind`, `rt`, `all`, or `preflight`.

## 2. Wind forecaster

Original V3.3 source: `forecast_release/ll_wind_xgb/train_direct_wind_xgb.py`.
Development wind: `forecast_release/source_releases/horse_creek_forecasting_release/data/wind_5min_development_2025Dec_2026Jan.csv`.

- 48 observed 5-min increments -> 48 forecast increments (4 hours).
- XGBoost native 48-output trees, 100 trees, seed 47.
- Training: 5 Dec 2025 to 20 Jan 2026; validation: 21–28 Jan 2026.
- Missing wind observations must not be interpolated into training samples.
- Outputs: `training_runs/xgboost_wind/direct_xgb_wind.json`,
  `metrics.json`, `by_horizon.csv`, `TRAINING_MANIFEST.json`.
- Runtime loader continues using its frozen `forecast_release/ll_wind_xgb/direct_xgb_wind.json`.

## 3. RT15 price forecaster

Original V3.3 development script: `training/xgboost/forecast_rt15_spread.py`.
This script fits *two* variants: a DA/wind-only model and an RT-history
model. The **RT-history** `XGB_RT_spread_development_2025.json` is compatible
with the active low-level price runtime's 31-feature interface, and is the
candidate corresponding to its frozen model. The historical script exports
validation/test metrics and comparison plots as well.

- Train Jan–Jun 2025; validate Jul–Aug 2025; remaining Sep–Dec 2025
  is development confirmation (not a pristine out-of-sample test).
- 15-min rolling issue, 16 x 15-min horizon (4 hours); target is RT minus DA.
- Uses cleared DA forecasts, latest completed hourly wind, causal RT lag/spread
  features and calendar variables, subject to a publication timing assumption.
- The frozen baseline `xgb_zero_delay_2025_trainJanJun.json` assumes the last
  completed RT settlement interval is immediately available at next quarter hour.
  This is a **simulation assumption**, not verified ERCOT publication timing.
- `--publication-lag 0` matches the original checkpoint specification; using
  e.g. `--publication-lag 60` trains a distinct delay-aware model and does not
  reproduce the frozen evaluation setup. Model fitting may still differ across
  XGBoost versions and environments.
- Output in `training_runs/xgboost_rt15/` is **not promoted** automatically.

## 4. Reproducibility boundaries

The shipped code is the V3.3 historical training implementation, but full
training has not been rerun as part of the repository consolidation. Inference
schema and checkpoint architecture can be verified with `preflight`; that
**does not prove the source recreates identical model weights**.
The 2026 holdout windows are never included in these development scripts,
except predeployment Jan 2026 wind samples used for wind training/validation.
Keep run manifests, library versions and evaluation metrics alongside any
newly trained checkpoints.
