"""Run ALL NINE matched deterministic HL baselines with continuous CVAE v3.

Spyder: open this file in the extracted HC_MPC_V3 folder, press F5.
Run VERIFY_HOLDOUT_PACKAGE_FROM_SPYDER.py first for environment checks.

Nothing in the original 5-min LL optimizer, SoC handover or existing 2026
stochastic experiments is changed. Each issue still produces 500 continuous
raw CVAE trajectories and reduces to 50 weighted paths over the 84h HL interval.
Only for the HL solver, 50 weighted paths are collapsed to ONE expected path.
Separate output directory; 15 local-day replay, exact seven UTC scored days.
Run after all code in this archive is EXTRACTED (not directly from the ZIP).
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

# Frozen controls: SAME as RUN_ALL_HOLDOUTS_FROM_SPYDER.py.
# No need to copy/modify/change the stochastic script or solver source files.
import RUN_ALL_HOLDOUTS_FROM_SPYDER as stochastic
from deterministic_hl_baseline import CaseLogger
from holdout_runtime import SEASONS

DETERMINISTIC_RESULTS_DIR = HERE/'results'/'HOLDOUT_2026_3SEASONS_3SYSTEMS_alpha1_CONTINUOUS_V3_DETERMINISTIC_HL_WEIGHTED_MEAN'
# Optional: run a subset when testing; for the *paper* leave this full 3x3.
# Completed cases are automatically skipped on rerun.


def run_all():
    import run_multiday as ptes_runner
    import wind_only.runner as wind_runner
    orig_ptes_run = ptes_runner.run
    orig_wind_run = wind_runner.run
    orig_ptes_solve = ptes_runner.solve_hl_market
    orig_wind_solve = wind_runner.solve_wind_hl
    orig_base_settings = stochastic.settings
    orig_base_hashes = stochastic._source_hashes
    original_result_dir=stochastic.RESULTS_DIR
    current = {'case_logger': None}
    if getattr(orig_ptes_solve, '_deterministic_wrapper', False):
        raise RuntimeError('Deterministic launcher already installed in this Python process. Restart interpreter before a new full run.')

    def _ptes_hl(wind_scenarios, da_scenarios, weights,
                 initial_energy_mwhth, rt_spread_hourly, risk_premium_hourly, **kwargs):
        ctx=current['case_logger']
        if ctx is None or ctx.case_name=='wind_only':
            raise AssertionError('PTES HL solver invoked without matching deterministic-case context')
        w, p, pi=ctx.record(wind_scenarios, da_scenarios, weights)
        return orig_ptes_solve(w,p,pi,initial_energy_mwhth,
                               rt_spread_hourly,risk_premium_hourly,**kwargs)

    def _wind_hl(wind_scenarios, da_scenarios, weights, spread_hourly,risk_hourly,**kwargs):
        ctx=current['case_logger']
        if ctx is None or ctx.case_name!='wind_only':
            raise AssertionError('Wind HL solver invoked without matching deterministic-case context')
        w,p,pi=ctx.record(wind_scenarios,da_scenarios,weights)
        return orig_wind_solve(w,p,pi,spread_hourly,risk_hourly,**kwargs)

    _ptes_hl._deterministic_wrapper=True
    _wind_hl._deterministic_wrapper=True

    def _with_context(runner,case,**kwargs):
        dest=Path(kwargs['out'])
        if dest.parent.name not in stochastic.SEASON_ORDER or dest.name!=case:
            raise AssertionError('Unexpected deterministic case output destination')
        if current['case_logger'] is not None:
            raise RuntimeError('Nested deterministic case runner')
        first_day=SEASONS[dest.parent.name][0]
        if first_day is None:raise AssertionError('No first market day')
        # Incomplete case reruns start at the fixed episode initial state.
        # Clear ONLY their derived deterministic audit CSVs so an interrupted
        # earlier attempt cannot leave duplicate issuance records.
        if not (dest/'SCORED_UTC_SUMMARY.json').exists():
            for derived in ('DETERMINISTIC_HL_INPUT_AUDIT.csv','DETERMINISTIC_HL_MEAN_INPUTS.csv'):
                (dest/derived).unlink(missing_ok=True)
        ctx=CaseLogger(dest,first_day,case)
        current['case_logger']=ctx
        try:
            out=runner(**kwargs)
            if ctx.count!=15:
                raise AssertionError(f'{dest}: expected 15 daily HL issues, got {ctx.count}')
            return out
        finally:
            current['case_logger']=None

    def _ptes_run(**kwargs):
        return _with_context(orig_ptes_run,Path(kwargs['out']).name,**kwargs)

    def _wind_run(**kwargs):
        return _with_context(orig_wind_run,'wind_only',**kwargs)

    def _settings():
        d=orig_base_settings()
        d.update(hl_policy='deterministic_probability_weighted_mean_of_v3_reduced_50',
                 hl_solver_scenarios=1,forecast_retained_scenarios=50,
                 forecast_draws=500,
                 deterministic_mean_reduction_window_hours=84,
                 deterministic_runner='RUN_DETERMINISTIC_HOLDOUTS_FROM_SPYDER.py',
                 comparison='same calibrated continuous CVAE v3 forecast and LL as stochastic 3x3; no retuning')
        return d

    def _hashes():
        d=orig_base_hashes()
        for name in ('RUN_DETERMINISTIC_HOLDOUTS_FROM_SPYDER.py','deterministic_hl_baseline.py'):
            d[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        return d

    # Exact same simulation code and data; wrappers intercept only the HL
    # solver call. The existing wind-only/ptes market runners still pass 50
    # selected scenarios for the causal SoC rollout and their forecast logs.
    ptes_runner.solve_hl_market=_ptes_hl
    wind_runner.solve_wind_hl=_wind_hl
    ptes_runner.run=_ptes_run
    wind_runner.run=_wind_run
    stochastic.RESULTS_DIR=DETERMINISTIC_RESULTS_DIR
    stochastic.settings=_settings
    stochastic._source_hashes=_hashes
    print('RUNNING DETERMINISTIC HL: 500 full CVAE v3 paths -> 50 weighted representatives -> 1 weighted mean HL path.',flush=True)
    print('ALL physical/LL/SoC/settlement settings unchanged. Output:',DETERMINISTIC_RESULTS_DIR,flush=True)
    try:
        results=stochastic.run_all()
        for season in stochastic.SEASON_ORDER:
            for case in stochastic.CASE_ORDER:
                p=DETERMINISTIC_RESULTS_DIR/season/case
                for name,expected in [('DETERMINISTIC_HL_INPUT_AUDIT.csv',15),
                                      ('DETERMINISTIC_HL_MEAN_INPUTS.csv',15*84)]:
                    file=p/name
                    if not file.is_file():
                        raise AssertionError(f'Missing audit file for {season}/{case}: {file}')
                    import pandas as pd
                    if len(pd.read_csv(file))!=expected:
                        raise AssertionError(f'Wrong audit rows in {file}')
        (DETERMINISTIC_RESULTS_DIR/'DETERMINISTIC_BASELINE_COMPLETED.json').write_text(json.dumps({
            'hl_policy':'single expected weighted trajectory of same 50 v3 reduced scenarios',
            'forecast_raw_n':500,'forecast_reduced_n':50,'hl_scenarios':1,
            'scored_case_count':9,'state':'complete, every case has 15 logged HL issues',
            'original_stochastic_files_modified':False,
        },indent=2),encoding='utf-8')
        print('DETERMINISTIC 3x3 COMPLETE: all nine case summaries and 135 daily HL audits verified.',flush=True)
        print('For optional comparison to the previous stochastic v3 results, run COMPARE_V3_DETERMINISTIC_TO_STOCHASTIC.py.',flush=True)
        return results
    finally:
        ptes_runner.solve_hl_market=orig_ptes_solve
        wind_runner.solve_wind_hl=orig_wind_solve
        ptes_runner.run=orig_ptes_run
        wind_runner.run=orig_wind_run
        stochastic.RESULTS_DIR=original_result_dir
        stochastic.settings=orig_base_settings
        stochastic._source_hashes=orig_base_hashes

if __name__ == '__main__':
    run_all()
