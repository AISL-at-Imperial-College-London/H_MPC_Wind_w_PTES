# Horse Creek wind–PTES: hierarchical model predictive control

A research codebase for studying how **pumped thermal energy storage (PTES)** can be operated alongside a wind farm participating in day-ahead (DA) and real-time (RT) electricity markets. The case study uses historical wind-generation and electricity-price data from the Horse Creek wind farm / ERCOT market in Texas. The co-located PTES plant is **hypothetical**; this is a historical closed-loop simulation, not a record of PTES operation at Horse Creek.

The repository provides the simulation code, local evaluation inputs, frozen forecasting models, and scripts to train the CVAE and XGBoost forecasters. It supports both a **commercial Gurobi solver** and an **open-source HiGHS alternative**.

## What the study does

The controller has two levels:

1. **Daily high-level MPC (HL-MPC)** constructs joint wind and DA-price scenarios and chooses hourly DA energy commitments and a storage-inventory reference. Two policies can be compared: stochastic optimisation over 50 weighted scenarios, and deterministic optimisation over their probability-weighted mean.
2. **Five-minute low-level MPC (LL-MPC)** uses short-term wind and RT-price forecasts to dispatch wind and PTES, subject to plant operating limits and a shared point-of-interconnection constraint. DA delivery deviations are settled using historical RT prices.

A separately optimised **wind-only** controller and a **half-duration PTES** variant serve as comparisons. Only storage *energy capacity* is halved; charging and discharging power ratings are unchanged. Results report DA + RT electricity settlement revenue; operating, degradation, and financing costs are not modelled.

## Start here: installation and solver choice

Requires **Python 3.11 or newer**. Commands below are run from the extracted repository directory (the directory containing `run.py`). A virtual environment is recommended.

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Linux/macOS:         source .venv/bin/activate
python -m pip install -r requirements-highs.txt
```

**No Gurobi licence is necessary** for the HiGHS option. SciPy provides the HiGHS MILP solver. Check the installation, models and data:

```bash
python run.py preflight --solver highs
python run.py solver-check
```

Then select a study:

```bash
# Main out-of-sample study: 3 weeks × 3 systems × stochastic HL
python run.py holdouts --solver highs

# Matched deterministic-HL study: 3 weeks × 3 systems
python run.py deterministic --solver highs

# Additional historical/development weeks, both HL policies: 18 cases
python run.py extra --solver highs
```

**To run the original Gurobi configuration instead**, install its optional dependency and activate a suitable Gurobi licence:

```bash
python -m pip install -r requirements-gurobi.txt
python run.py preflight
python run.py holdouts
python run.py deterministic
```

The **default solver is Gurobi**, to preserve the original experiment configuration. Always specify `--solver highs` when you intend to run without it. `--dry-run` describes any of the three campaigns without running a simulation, e.g. `python run.py holdouts --solver highs --dry-run`.

> **Solver-equivalence limit:** HiGHS solves the high-level MILP and the wind-only models, but the PTES low-level problem uses a **piecewise-linear MILP approximation** of the quadratic terminal-inventory penalty instead of Gurobi's exact MIQP. Feasible trajectories and revenues can therefore differ. HiGHS runs are stored separately under folders ending `_HIGHS`, never mixed with the Gurobi results. Full 50-scenario, multiweek HiGHS campaign performance has not been established; expect substantial computation.

## Available experiments

| Experiment | Scored UTC period | System configurations | HL policies |
|---|---|---|---|
| Primary holdout study | 9–15 February, 11–17 May, 13–19 July 2026 | Wind-only, PTES 50%, PTES 100% | Stochastic; deterministic run separately |
| Supplementary historical study | 8–14 December 2025, 4–10 May, 6–12 July 2026 | Same three configurations | Both |

Each primary case uses an initial warm-up period before the **seven-day** scored UTC interval. The supplementary periods are **not additional independent holdouts** and should not be pooled into claims about out-of-sample performance. Historical wind is treated as available generation, rather than measured PTES-coordinated dispatch.

All simulation output is written to `results/` (not included in Git). Completed cases have scored-period summary files; case-level forecast, solver and dispatch diagnostics are available alongside them. The existing scripts support resuming at case boundaries, subject to matching experiment fingerprints. See [experiment instructions](docs/EXPERIMENTS_AND_SOLVERS.md) for case definitions, output files and solver limitations.

## Forecast model training

Training scripts and the development data required by those scripts are included. Retraining is **separate from evaluation**: new checkpoints go to `training_runs/` and do not automatically replace the frozen models used by the MPC runs.

| Model | Role | Re-training command |
|---|---|---|
| Continuous conditional VAE | Joint wind / DA-price scenario generator used by the MPC | `python train_forecasts.py continuous --epochs 80` |
| Thesis T-CVAE | Alternative 96 h / 96 h scenario model, not the active checkpoint | `python train_forecasts.py thesis --epochs 250` |
| Wind XGBoost | Five-minute wind forecasts | `python train_xgboost.py wind` |
| RT-price XGBoost | Short-term RT price-spread forecasts | `python train_xgboost.py rt --publication-lag 0` |

After continuous-CVAE training, run `python train_forecasts.py calibrate` to construct its calibration residual paths. Run `python train_xgboost.py preflight` to check XGBoost training inputs. See the [forecast-training guide](training/README.md) and [XGBoost guide](training/xgboost/README.md) for data splits, hyperparameters and model compatibility.

The RT-price model's zero-minute publication-lag setting is a **modelling assumption**, not independently validated market publication timing. The thesis model and the continuous MPC model are different architectures and their checkpoints must not be exchanged.

## Repository map

| Path | Purpose |
|---|---|
| `run.py` | Public command-line entry point for verification and experiments |
| `solver_campaign.py` | Select HiGHS/Gurobi and isolate alternative-solver results |
| `ptes_mpc/` | PTES physics, HL/LL optimisation, dispatch and settlement |
| `wind_only/` | Wind-only baseline optimiser and simulation |
| `run_multiday.py`, `holdout_runtime.py`, `extra_v3_runtime.py` | Closed-loop historical replay and local-market data adapters |
| `forecast_release/` | Frozen CVAE/XGBoost models, inference and source training routines |
| `holdout_inputs/` | Historical evaluation inputs for six selected weeks |
| `train_forecasts.py`, `train_xgboost.py`, `training/` | Training entry points and documentation |
| `tests/` | Regression and solver-independent scientific checks |
| `docs/` | Detailed experiment, solver and provenance notes |

For users of **Spyder on Windows**, the `RUN_*_FROM_SPYDER.py` programs remain available. These original runners are configured for Gurobi. Users without Gurobi should run the documented `python run.py ... --solver highs` commands in a terminal (including the Spyder/IPython terminal, if configured).

## Reproducibility and distribution

Run `python run.py tests` for the regression suite. The `SOURCE_MANIFEST.json` lists checksums for the original retained research files, models and datasets. The training and execution wrappers do not automatically promote new checkpoints or overwrite the frozen scientific source.

**Scope of validation:** HiGHS high-level and low-level smoke tests have passed. The full historical campaigns have **not** been rerun in this packaged environment. Installing the code and passing the preflight does not establish numerical replication of the manuscript results, which also depend on solver versions, tolerances, forecasting checkpoints and initial-inventory handling.

**Publication notice:** No redistribution licence accompanied the supplied code/data. Before making a public release, establish permission to distribute the market and wind-generation datasets, confirm model/data provenance, and add a code licence and publication citation. A private repository is the safer starting point.
