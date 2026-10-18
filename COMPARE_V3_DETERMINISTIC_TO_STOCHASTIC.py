"""Matched 2026 v3 deterministic vs stochastic scoreboard and forecast-parity QA.

Spyder: run after RUN_DETERMINISTIC_HOLDOUTS_FROM_SPYDER.py completes.
The companion stochastic_v3_reference/ data are extracted from the uploaded
results(6).zip; they are NOT recomputed or forecast/inference inputs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import json

ROOT=Path(__file__).resolve().parent
DET=ROOT/'results'/'HOLDOUT_2026_3SEASONS_3SYSTEMS_alpha1_CONTINUOUS_V3_DETERMINISTIC_HL_WEIGHTED_MEAN'
REF=ROOT/'stochastic_v3_reference'
SEASONS=('winter','spring','summer')
CASES=('wind_only','ptes_50pct','ptes_100pct')


def compare():
    if not (DET/'COMPLETED_ALL_NINE.json').exists():
        raise FileNotFoundError('Deterministic 3x3 not finished. Run RUN_DETERMINISTIC_HOLDOUTS_FROM_SPYDER.py first.')
    det=pd.read_csv(DET/'ALL_9_CASES_COMPARISON.csv')
    sto=pd.read_csv(REF/'STOCHASTIC_V3_ALL_NINE_SCORED.csv')
    if set(map(tuple,det[['season','case']].to_numpy())) != set(map(tuple,sto[['season','case']].to_numpy())):
        raise AssertionError('Different case sets; cannot make matched 3x3 comparison')
    joined=sto.merge(det,on=['season','case'],suffixes=('_stochastic','_deterministic'),validate='1:1')
    out=pd.DataFrame({'season':joined.season,'case':joined.case,
         'stochastic_v3_revenue_usd':joined.total_electricity_revenue_usd_stochastic,
         'deterministic_v3_revenue_usd':joined.total_electricity_revenue_usd_deterministic,
         'stochastic_minus_deterministic_usd':joined.total_electricity_revenue_usd_stochastic-joined.total_electricity_revenue_usd_deterministic,
         'stochastic_minus_deterministic_percent':100*(joined.total_electricity_revenue_usd_stochastic-joined.total_electricity_revenue_usd_deterministic)/joined.total_electricity_revenue_usd_deterministic,
         'stochastic_open_soc_mwhth':joined.initial_scored_soc_mwhth_stochastic,
         'deterministic_open_soc_mwhth':joined.initial_scored_soc_mwhth_deterministic,
         'stochastic_close_soc_mwhth':joined.final_soc_mwhth_stochastic,
         'deterministic_close_soc_mwhth':joined.final_soc_mwhth_deterministic,
         'stochastic_curtailed_mwh':joined.wind_curtailed_mwh_stochastic,
         'deterministic_curtailed_mwh':joined.wind_curtailed_mwh_deterministic,
         'stochastic_abs_DA_deviation_mwh':joined.abs_quarter_DA_deviation_mwh_stochastic,
         'deterministic_abs_DA_deviation_mwh':joined.abs_quarter_DA_deviation_mwh_deterministic})
    # IMPORTANT: verify the FULL 84h stochastic weighted mean is exactly the
    # forecast handed into the deterministic solver. Wind-only has 24h published
    # forecasts because no inventory spans the remaining 60h.
    ref=pd.read_csv(REF/'STOCHASTIC_V3_HL_ISSUED_MEANS.csv')
    det_means=[]
    for season in SEASONS:
        for case in CASES:
            f=DET/season/case/'DETERMINISTIC_HL_MEAN_INPUTS.csv'
            if not f.exists():raise FileNotFoundError(f'Expected complete audit: {f}')
            d=pd.read_csv(f)
            if len(d)!=15*84:raise AssertionError(f'Incorrect 15x84 deterministic forecast output for {season}/{case}')
            d.insert(0,'case',case);d.insert(0,'season',season)
            det_means.append(d)
    det_all=pd.concat(det_means,ignore_index=True)
    # Original model log uses `YYYY-MM-DD HH:MM:SS+00:00`, while the new
    # audit uses ISO8601 `YYYY-MM-DDTHH:MM:SS+00:00`. Match UTC instants,
    # not their string representations.
    ref['hour_start_utc']=pd.to_datetime(ref['hour_start_utc'],utc=True)
    det_all['hour_start_utc']=pd.to_datetime(det_all['hour_start_utc'],utc=True)
    merged=ref.merge(det_all,on=['season','case','delivery_day_local','hour_start_utc'],validate='1:1',suffixes=('_stoch','_det'))
    if len(merged)!=len(ref):raise AssertionError('Missing or mismatched stochastic reference forecast timestamps')
    err_w=np.abs(merged.expected_wind_mw.to_numpy()-merged.deterministic_wind_mean_mw.to_numpy())
    err_p=np.abs(merged.expected_DA_price_usd_mwh.to_numpy()-merged.deterministic_DA_price_mean_usd_mwh.to_numpy())
    # CPU/GPU/PyTorch floating-point kernels can differ at the 1e-5 level
    # even with fixed seed. Tolerance of 0.01 MW / 0.01 USD-MWh is far below
    # forecast uncertainty, but still catches inadvertent forecast swapping.
    tol_w=0.01;tol_p=0.01
    q=dict(n_checked_forecast_hours=len(ref),max_wind_mw_abs_difference=float(err_w.max()),
           max_da_usd_mwh_abs_difference=float(err_p.max()),
           wind_MW_tolerance=tol_w,DA_price_USD_MWh_tolerance=tol_p,
           forecast_means_match=bool(err_w.max()<tol_w and err_p.max()<tol_p))
    (DET/'MATCHED_V3_FORECAST_MEAN_CHECK.json').write_text(json.dumps(q,indent=2))
    if not q['forecast_means_match']:
        raise AssertionError('v3 weighted forecast means DO NOT MATCH previous stochastic run; comparison is not controlled. See MATCHED_V3_FORECAST_MEAN_CHECK.json')
    out.to_csv(DET/'MATCHED_V3_STOCHASTIC_VS_DETERMINISTIC_SCORED.csv',index=False)
    print('\nMATCHED continuous CVAE v3 policies (USD; the exact seven-day scored UTC windows):')
    print(out[['season','case','stochastic_v3_revenue_usd','deterministic_v3_revenue_usd',
               'stochastic_minus_deterministic_usd','stochastic_minus_deterministic_percent']].to_string(index=False))
    print('\nForecast parity check: ',q)
    return out

if __name__=='__main__':
    compare()
