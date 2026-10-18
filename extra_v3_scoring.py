"""Extra-week UTC financial scoring reused verbatim from the original 3 Extra Weeks runner.

The V3 protected-holdout scorer requires >6 days warm-up and cannot be applied
unchanged to the 42h/67h historical supplemental replays. This scoring
implementation preserves the original extra-week checks (>24h warm-up).
"""
from __future__ import annotations
import json
import pandas as pd
import numpy as np
WIND_RAMP_MW_PER_5MIN=40.0

def v_warm_start(active):
    return pd.Timestamp(active['first_market_day']).tz_localize('America/Chicago').tz_convert('UTC')


def _scored_financials(folder,active,case):
    efile=folder/('five_minute_wind_only.csv' if case=='wind_only' else 'five_minute_closed_loop.csv')
    qfile=folder/('da_commitments_wind_only.csv' if case=='wind_only' else 'da_commitments.csv')
    e=pd.read_csv(efile);q=pd.read_csv(qfile)
    e['time_utc']=pd.to_datetime(e['time_utc'],utc=True,format='mixed')
    q['hour_start_utc']=pd.to_datetime(q['hour_start_utc'],utc=True,format='mixed')
    a=active['score_start'];b=active['score_end']
    warm=e[e.time_utc<a]
    v=e[(e.time_utc>=a)&(e.time_utc<b)].copy().sort_values('time_utc')
    times=pd.date_range(a,b-pd.Timedelta(minutes=5),freq='5min',tz='UTC')
    if len(v)!=2016 or not v.time_utc.reset_index(drop=True).equals(pd.Series(times)):
        raise AssertionError(f'{case}: score must be exactly 2016 consecutive 5-min UTC records')
    if len(warm)<288:
        raise AssertionError('Insufficient causal warm-up observations (at least 24 h needed)')
    q=q.set_index('hour_start_utc')['DA_commitment_MW'].sort_index()
    hindex=pd.date_range(a,b-pd.Timedelta(hours=1),freq='h',tz='UTC')
    qq=q.reindex(hindex)
    da=active['hourly']['DA_price_usd_mwh'].reindex(hindex)
    if qq.isna().any() or da.isna().any():raise AssertionError('Missing scored DA bids or clearing prices')
    q15=qq.reindex(v.time_utc.dt.floor('h')).to_numpy(dtype=float)
    v['DA_commitment_MW']=q15
    v['DA_USD_per_MWh']=active['hourly']['DA_price_usd_mwh'].reindex(v.time_utc.dt.floor('h')).to_numpy(float)
    # Per-five-minute exact algebra equivalent to 15-minute signed settlement.
    v['RT_price_USD_per_MWh']=active['rt'].reindex(v.time_utc.dt.floor('15min')).to_numpy(float)
    if v.RT_price_USD_per_MWh.isna().any():raise AssertionError('Missing scored RT settlement price')
    v['DA_cash_USD']=v.DA_commitment_MW*v.DA_USD_per_MWh/12.
    v['RT_cash_USD']=(v.physical_poi_mw-v.DA_commitment_MW)*v.RT_price_USD_per_MWh/12.
    v['electricity_cash_USD']=v.DA_cash_USD+v.RT_cash_USD
    v['wind_curtailment_mwh']=v.curtailed_wind_mw/12
    v['net_poi_mwh']=v.physical_poi_mw/12
    v['utc_day']=v.time_utc.dt.strftime('%Y-%m-%d')
    energy_qtr=v.groupby(v.time_utc.dt.floor('15min')).apply(
        lambda block:float((block.physical_poi_mw-block.DA_commitment_MW).sum()/12.),include_groups=False)
    if len(energy_qtr)!=672:raise AssertionError('Expected exactly 672 settlement quarters')
    v['RT_energy_dev_mwh']=(v.physical_poi_mw-v.DA_commitment_MW)/12.
    values=dict(
        season=active['season'],case=case,storage_fraction={'wind_only':0,'ptes_50pct':.5,'ptes_100pct':1.}[case],
        score_start_utc=a.isoformat(),score_end_utc=b.isoformat(),
        scored_5min_steps=2016,scored_hours=168,scored_quarters=672,
        da_revenue_usd=float(v.DA_cash_USD.sum()),rt_revenue_usd=float(v.RT_cash_USD.sum()),
        total_electricity_revenue_usd=float(v.electricity_cash_USD.sum()),
        da_sold_mwh=float(qq.sum()),available_wind_mwh=float((v.available_wind_mw if case=='wind_only' else v.actual_wind_t0_mw).sum()/12),
        wind_delivered_mwh=float(v.dispatched_wind_mw.sum()/12),
        wind_curtailed_mwh=float(v.curtailed_wind_mw.sum()/12),
        net_poi_mwh=float(v.physical_poi_mw.sum()/12),
        abs_quarter_DA_deviation_mwh=float(energy_qtr.abs().sum()),
        final_soc_mwhth=(None if case=='wind_only' else float(v.tank_mwhth.iloc[-1])),
        initial_scored_soc_mwhth=(None if case=='wind_only' else float(warm.tank_mwhth.iloc[-1])),
        ptes_charge_mwh=(0. if case=='wind_only' else float(v.charge_mw.sum()/12)),
        ptes_discharge_mwh=(0. if case=='wind_only' else float(v.discharge_mw.sum()/12)),
        safety_interventions=(0 if case=='wind_only' else int(v.safety_override.sum())),
        wind_up_ramp_exceedances=int((v.dispatched_wind_mw.diff()>WIND_RAMP_MW_PER_5MIN+1e-4).sum()),
        ll_solver_failures=0,
        filled_wind_timestamps=active['wind_repairs'],filled_rt_timestamps=active['rt_repairs'],
        actual_warmup_hours=float((a-v_warm_start(active))/pd.Timedelta(hours=1)),
        episode_data_status='non_holdout_reused_development_or_warmup',
        post_score_simulated_tail_excluded=True)
    if abs(values['da_revenue_usd']-float((qq*da).sum()))>1e-5:raise AssertionError('DA revenue mismatch')
    if abs(values['da_revenue_usd']+values['rt_revenue_usd']-values['total_electricity_revenue_usd'])>1e-4:
        raise AssertionError('DA/RT revenue does not reconcile')
    v.to_csv(folder/'SCORED_UTC_5MIN.csv',index=False)
    pd.DataFrame({'quarter_start_utc':energy_qtr.index,'energy_deviation_mwh':energy_qtr.values,
                  'rt_settlement_price_usd_mwh':active['rt'].reindex(energy_qtr.index).to_numpy(float)}).to_csv(
                  folder/'SCORED_UTC_QUARTERS.csv',index=False)
    daily=[]
    for day,g in v.groupby('utc_day'):
        d=dict(season=active['season'],case=case,utc_day=day,
             DA_USD=float(g.DA_cash_USD.sum()),RT_USD=float(g.RT_cash_USD.sum()),
             revenue_USD=float(g.electricity_cash_USD.sum()),
             wind_available_mwh=float((g.available_wind_mw if case=='wind_only' else g.actual_wind_t0_mw).sum()/12),
             wind_curtailed_mwh=float(g.curtailed_wind_mw.sum()/12),
             POI_mwh=float(g.physical_poi_mw.sum()/12),
             end_soc_mwhth=(None if case=='wind_only' else float(g.tank_mwhth.iloc[-1])))
        daily.append(d)
    if len(daily)!=7 or abs(sum(x['revenue_USD'] for x in daily)-values['total_electricity_revenue_usd'])>1e-5:
        raise AssertionError('UTC daily metrics must reconcile to week')
    pd.DataFrame(daily).to_csv(folder/'SCORED_UTC_DAILY.csv',index=False)
    (folder/'SCORED_UTC_SUMMARY.json').write_text(json.dumps(values,indent=2,default=str))
    return values,daily

