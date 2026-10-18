# Experiments, solvers and reproducibility details

This document supplements the [main README](../README.md). The public entry point is `python run.py` from the repository root.

## Solver implementation

| Component | Gurobi | SciPy / HiGHS |
|---|---|---|
| HL PTES | Exact mixed-integer linear programme (MILP) | Same MILP formulation (potential numerical differences) |
| HL wind-only | LP/MILP | LP/MILP |
| LL PTES | Exact convex MIQP (linear constraints + quadratic terminal SoC penalty) | Mixed-integer linear programme with epigraph tangents approximating terminal SoC penalty |
| LL wind-only | Linear optimisation | Linear optimisation |

The PTES low-level HiGHS branch is implemented in `ptes_mpc/ll_mpc.py` and `ptes_mpc/linear_mip.py`; it is **not** a newly introduced numerical approximation. The repository's new `solver_campaign.py` exposes this already-existing backend to the original full-campaign scripts. The two backends need not reproduce identical decisions or objective values; solver tolerances, approximation error, time limits and solver success statuses should be checked before drawing conclusions.

### Why the original script names mention Spyder and Gurobi

Those scripts are snapshots of the manuscript evaluation programs. Their internal `HL_BACKEND` and `LL_BACKEND` settings both default to `gurobi`; the supplementary launcher also checks the Gurobi licence explicitly. The public `run.py` launcher changes those **in-memory** settings only when `--solver highs` is selected. Nothing in the preserved scientific modules or model checkpoint files is rewritten.

HiGHS runs are directed to `results/*_HIGHS` rather than the original directories. `--dry-run` prints destinations and the case count without optimising. The original Gurobi studies still run with their original launcher configuration and destination names.

## Recommended workflow

1. Install `requirements-highs.txt` for the open-source environment, or `requirements-gurobi.txt` (plus a licence) for the original setting.
2. Run `python run.py preflight --solver highs` and `python run.py solver-check` for an open-source installation.
3. Run `python run.py holdouts --solver highs` to generate stochastic results. The full stochastic HL has 50 reduced weighted scenarios and an 84-hour decision-relevant horizon. Heavy optimisation is expected.
4. Run `python run.py deterministic --solver highs` to compare against the probability-weighted mean of those same scenarios. The deterministic run is not a perfect-foresight benchmark.
5. Examine each case's scored summary and audit files. Compare only cases with matching weeks, data and solver settings. Keep HiGHS and Gurobi results separate.

For publication-level replication of the originally reported results, use the Gurobi mode and verify that the **same frozen forecasts, case settings, state initialisation and available historical observations** have been used. The default published-value target cannot be established simply by running the HiGHS alternative.

## Campaigns and folders

- `holdouts`: nine stochastic cases (winter, spring, summer × wind-only, half-duration PTES, standard PTES). There are 15 local-market-day cycles per case including warm-up and tail. Scores cover exactly seven UTC days (2,016 five-minute samples).
- `deterministic`: nine paired cases for the identical primary weeks, using the 50-scenario weighted mean at the HL optimiser boundary. LL and settlement remain unchanged.
- `extra`: eighteen historical/development cases (December 2025, spring/summer supplementary 2026), both HL policies. These are **not** independent holdouts.

Output files for a completed case normally include `SCORED_UTC_SUMMARY.json`, `SCORED_UTC_5MIN.csv`, `SCORED_UTC_DAILY.csv`, `SCORED_UTC_QUARTERS.csv` and the relevant DA/LL forecast and dispatch logs. A campaign `FROZEN_EXPERIMENT_PROTOCOL.json` captures settings and core source/data/checkpoint hashes. Interrupted case execution may leave partial outputs; a case can be restarted through the original campaign logic when its fingerprint is unchanged.

The evaluation objective is historical DA+RT electricity-settlement revenue with signed RT deviations. O&M, fuel, degradation and capital charges are **not** part of the reported economic objective. Any comparison across runs should also inspect warm-up inventory and scored initial/terminal thermal inventories before interpreting a revenue difference as a storage benefit.

## Data scope and model compatibility

`holdout_inputs/` contains historical wind generation and prices needed for the six weeks, plus context used by forecasting and simulation. The model-development CSVs under `forecast_release/source_releases/horse_creek_forecasting_release/data/` supply the CVAE and XGBoost training workflows. The repository does not contain the complete raw acquisition/download archive or past large simulation-output directories.

The historical data are supplied for research reproducibility, subject to redistribution rights. Trained model checkpoints are frozen for the evaluation. Newly trained checkpoints are saved to `training_runs/` and must be explicitly validated for conditioning continuity, predictive skill, data chronology and parameter compatibility before use in a paper-level comparison.

## Verification levels

- **Preflight:** verifies data coverage, imported modules/checkpoints and scenario generation; with `--solver highs` it does not check a Gurobi licence.
- **Solver-check:** exercises HiGHS high-level wind-only and PTES models and a low-level PTES smoke test; not a seven-day replay.
- **Regression tests:** `python run.py tests` exercises unit and behaviour checks.
- **Full campaign:** evaluates causal simulated trajectories, market commitments and revenue over historic weeks. This last level was **not** completed as part of the repository packaging process.
