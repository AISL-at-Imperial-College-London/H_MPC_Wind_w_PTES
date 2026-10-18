#!/usr/bin/env python3
"""Chronological real-data integration rehearsal, NEVER a holdout score.

Default date Dec 18 2025 lies within frozen five-minute wind XGB training.
Outputs prove as-of forecast interfaces + 84h HL + 5-min LL + independent
physical replay and signed ERCOT-style two-settlement cashflow. This is an
engineering diagnostic, not an independent performance demonstration.
"""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
import pandas as pd
from ptes_mpc.hl_market import solve_hl_market,estimate_spread_from_past
from ptes_mpc.physics import PlantParams,PlantState
from ptes_mpc.harness import simulate_5min
from ptes_mpc.adapters import FrozenProviders

HERE=Path(__file__).resolve().parent
ROOT=HERE/'forecast_release'
DATA=ROOT/'source_releases'/'horse_creek_forecasting_release'/'data'
FORECAST=ROOT/'source_releases'/'horse_creek_forecasting_release'
CHICAGO='America/Chicago'

def load_raw():
    hourly=pd.read_csv(DATA/'development_master_hourly_2018_2021_2024_2025.csv',
                       parse_dates=['timestamp_utc']).set_index('timestamp_utc').sort_index()
    rt=pd.read_csv(DATA/'rt15_all_development_2018_2021_2025.csv',
                   parse_dates=['timestamp_utc']).set_index('timestamp_utc')['RT15_price_usd_mwh']
    w=pd.read_csv(DATA/'wind_5min_development_2025Dec_2026Jan.csv',
                  parse_dates=['timestamp_utc']).set_index('timestamp_utc')['HorseCreek_MW']
    return hourly,rt,w

def clearing_asof(issue_utc):
    """Conservative UNVERIFIED study convention: previous local day 18:00,
    when issuing forecasts before 18:00; otherwise current local day 18:00.
    NOT an ERCOT archived published_at timestamp. No future DA revealed.
    """
    local=pd.Timestamp(issue_utc).tz_convert(CHICAGO)
    last=local.normalize()+pd.Timedelta(hours=18)
    if last>local:last-=pd.Timedelta(days=1)
    return last.tz_convert('UTC')

def available_da(issue_utc,all_da):
    # Model cannot see next-day DA until after assumed clearing time.
    local=pd.Timestamp(issue_utc).tz_convert(CHICAGO)
    latest=local.date() if local.hour<18 else (local+pd.Timedelta(days=1)).date()
    mask=(all_da.index.tz_convert(CHICAGO).date<=latest)
    return all_da.loc[mask]

def soc_lookup(day_start, init, refs):
    """Interpolate a 1-hour HL SoC reference using RELATIVE elapsed hours.

    Never mix DatetimeIndex.asi8 and Timestamp.value: the former may be
    microsecond-based in pandas 3 while Timestamp.value is nanoseconds.
    Such a mismatch silently returns the last 84-hour reference for every
    bid-to-delivery lookup. Time deltas avoid pandas' internal units entirely.
    
    E[0] = initial inventory at HL delivery midnight;
    E[h+1] = predicted inventory after hour h, for h=0..83.
    Endpoint extrapolation matches the legacy interpolation policy.
    """
    start = pd.Timestamp(day_start)
    if start.tzinfo is None:
        raise ValueError('HL SoC reference requires a timezone-aware start')
    start = start.tz_convert('UTC')
    ref = np.asarray(refs, dtype=float).reshape(-1)
    vals = np.concatenate(([float(init)], ref))
    if len(vals) < 2 or not np.isfinite(vals).all():
        raise ValueError('HL SoC reference is empty or nonfinite')
    grid = np.arange(len(vals), dtype=float)

    def lookup(t):
        instant = pd.Timestamp(t)
        if instant.tzinfo is None:
            raise ValueError('SoC lookup time must be timezone-aware')
        elapsed_hours = (instant.tz_convert('UTC') - start) / pd.Timedelta(hours=1)
        return float(np.interp(float(elapsed_hours), grid, vals))

    return lookup

def baseline_wind_only(da,rt,wind,start,n_steps,qs):
    """Same-DA-schedule physical comparator, NOT optimal wind-only DA bids."""
    intervals=pd.date_range(start,periods=n_steps,freq='5min',tz='UTC')
    delivery=wind.reindex(intervals)
    if delivery.isna().any():raise ValueError('missing wind-only actuals')
    e=delivery.groupby(delivery.index.floor('15min')).sum()/12
    hours=sorted(set(intervals.floor('h')))
    da_cash=sum(float(da[h])*float(qs[h]) for h in hours)
    rt_cash=sum(float(rt[k])*(float(value)-float(qs[k.floor('h')])/4) for k,value in e.items())
    return {'same_DA_wind_only_total_usd':float(da_cash+rt_cash),
            'same_DA_wind_only_phys_mwh':float(delivery.sum()/12),
            'meaning':'Comparison uses identical HL PTES DA positions, not optimized wind-only bids'}

def run(*,day='2025-12-18',hours=2,ns=5,model='tcvae',
         start_fraction=0.,finish_fraction=None,backend='gurobi',hl_backend='gurobi',out=None,
         hl_time_limit=600.,ll_time_limit=30.,beta=1.,hl_gap=.01,ll_gap=.01,
         gurobi_threads=0,gurobi_seed=1,gurobi_output=False,
         quadratic_tangents=121):
    if day not in ('2025-12-18','2025-12-19','2025-12-20','2025-12-21'):
        raise ValueError('Phase 3 launcher restricted to known safe December 2025 engineering dates')
    if hours<1 or hours>24:raise ValueError('one-day episode 1..24 hours')
    if ns<2 or ns>50:raise ValueError('scenario count 2..50')
    if finish_fraction not in (None,0.,1.):raise ValueError('finish None,0,1')
    t0=time.time();prm=PlantParams()
    hourly,rt,wind=load_raw()
    day_start=pd.Timestamp(day).tz_localize(CHICAGO).tz_convert('UTC')
    # For these dates all local days have 24 hours; no DST or partial hour.
    scored_idx=pd.date_range(day_start,periods=hours*12,freq='5min',tz='UTC')
    past_wind=wind.loc[(wind.index>=day_start-pd.Timedelta(hours=12))&(wind.index<day_start)].dropna()
    actual_wind=wind.reindex(scored_idx)
    if actual_wind.isna().any():raise ValueError('actual 5-minute wind gap; no interpolation')
    da=hourly['DA_price_usd_mwh'].dropna()
    sys.path.insert(0,str(FORECAST))
    from horse_creek_forecasts import JointForecastProvider
    generated=JointForecastProvider(model).generate(day,n_raw=500,n_keep=ns,seed=3714)
    assert generated.timestamps_utc[0]==day_start
    spread,risk=estimate_spread_from_past(rt15=rt,da_hourly=da,
          decision_utc=generated.cutoff_utc,delivery_times_utc=generated.timestamps_utc)
    initial=PlantState(prm.capacity_mwhth*start_fraction,'off',6)
    hl=solve_hl_market(generated.wind_mw,generated.da_price_usd_mwh,generated.weights,
                      initial.energy_mwhth,spread,risk,time_limit_s=hl_time_limit,mip_rel_gap=hl_gap,
                      solver_backend=hl_backend,gurobi_threads=gurobi_threads,
                      gurobi_seed=gurobi_seed,gurobi_output=gurobi_output)
    if hl.commitment_mw is None:
        raise RuntimeError(f'HL has no feasible incumbent: {hl.message}')
    if not hl.success:
        raise RuntimeError(f'HL did not meet solver tolerance: {hl.message}')
    q=pd.Series(hl.commitment_mw,index=pd.date_range(day_start,periods=24,freq='h',tz='UTC'))
    lookup=soc_lookup(day_start,initial.energy_mwhth,hl.reference_mwhth)
    forecasts=FrozenProviders(ROOT,da_cleared_at_for=clearing_asof)
    def wind_predictor(t,completed_wind):
        return forecasts.wind(t,completed_wind)
    def rt_predictor(t,completed_rt,_all_da,completed_wind):
        historic_and_cleared=available_da(t,da)
        return forecasts.rt(t,completed_rt,historic_and_cleared,completed_wind)
    episode_target=(prm.capacity_mwhth*finish_fraction if finish_fraction is not None else None)
    clock=time.time()
    result=simulate_5min(start_utc=day_start,n_steps=hours*12,historic_wind_5min=past_wind,
            realized_wind_5min=actual_wind,realized_rt15_price=rt,
            da_commitments_hourly=q,da_prices_hourly=da,
            cleared_at_utc=clearing_asof(day_start),
            wind_predictor=wind_predictor,rt_predictor=rt_predictor,
            initial_state=initial,soc_target_at=lookup,prm=prm,
            episode_terminal_target_mwhth=episode_target,
            terminal_beta_usd_per_mwhth2=beta,
            solver_backend=backend,quadratic_tangents=quadratic_tangents,
            time_limit_s=ll_time_limit,mip_rel_gap=ll_gap,progress_every=12)
    elapsed_ll=time.time()-clock
    ev=result.events
    assert len(ev)==12*hours and ev['physical_poi_mw'].between(-230.0001,230.0001).all()
    assert abs(float(ev.physical_poi_mw.sum()/12)-result.cashflow['phys_mwh'])<1e-6
    assert result.cashflow['total_usd']==result.cashflow['da_usd']+result.cashflow['rt_usd']
    expected_refresh=hours*4
    assert ev.rt_forecast_refreshed_utc.nunique()==expected_refresh
    extras=baseline_wind_only(da,rt,wind,day_start,12*hours,q)
    summary={
       'phase':'chronological integration smoke test; NOT an independent holdout result',
       'date_local':day,'hours':hours,'five_minute_steps':hours*12,
       'warning':'WINDS DECEMBER 2025 OVERLAP FROZEN 5MIN XGBOOST TRAINING; DO NOT REPORT AS OOS ECONOMIC PERFORMANCE',
       'hl_scenarios':ns,'hl_raw_scenarios':500,'hl_scenario_generator':model,
       'hl_horizon_hours':84,'hl_solver_backend':hl.solver_backend,
       'hl_solver_success':hl.success,'hl_solver_message':hl.message,'hl_gap':hl.mip_gap,
       'hl_runtime_seconds':hl.runtime_seconds,'hl_best_bound_minimization':hl.best_bound,
       'hl_model_variables':hl.model_nvars,'hl_model_constraints':hl.model_nrows,
       'hl_firstday_DA_min_mw':float(q.min()),'hl_firstday_DA_max_mw':float(q.max()),
       'hl_firstday_DA_total_mwh':float(q.sum()),
       'expected_wind_firstday_mean_mw':float(np.dot(generated.weights,generated.wind_mw[:,:24]).mean()),
       'max_firstday_scenario_soc_difference_mwhth':float(np.ptp(hl.scenario_end_day1_mwhth)),
       'hl_rt_expectation':'past 28d local-hour RT-DA mean',
       'hl_deviation_risk':'past 28d local-hour RT-DA half-IQR; INTERNAL optimization proxy only',
       'hl_cutoff_utc':generated.cutoff_utc,'initial_soc_mwhth':initial.energy_mwhth,
       'finish_target_mwhth':episode_target,
       'final_soc_mwhth':result.final_state.energy_mwhth,
       'terminal_error_mwhth':result.actual_terminal_error_mwhth,
       'terminal_soft_penalty_usd_not_settlement':result.terminal_soft_penalty_usd,
       'actual_market_cashflows_usd':result.cashflow,
       'same_DA_wind_only_context':extras,
       'regulatory_interventions':result.regulatory_interventions,
       'forced_off_interventions':result.forced_off_interventions,
       'clipped_discharge_mwh':result.clipped_discharge_mwh,
       'max_abs_ptes_change_mw_per_5min':result.max_executed_ptes_power_ramp_mw_per_5min,
       'll_solver_backend':backend,'ll_target_gap':ll_gap,'ll_time_limit_s':ll_time_limit,
       'll_observed_gap_max':float(ev.mip_gap.max()),
       'll_solver_failures':result.solve_failures,'ll_compute_seconds':round(elapsed_ll,3),
       'total_compute_seconds':round(time.time()-t0,3),
       'settlement_convention':'signed simplified energy-only DA+RT two-settlement; no ancillaries or actual QSE invoice',
       'RT_availability':'immediately after completed quarter, simulated convention (not verified publication)',
       'DA_availability':'assume previous local day 18:00 clearing, unverified calendar proxy',
       'remaining_limitation':'full 50-scenario 84h Gurobi performance not validated in this environment; DA bid-to-delivery inventory and actual published_at data not validated'}
    folder=Path(out) if out else HERE/'results'/f'{day}_{hours}h_{ns}scenarios_{model}'
    folder.mkdir(parents=True,exist_ok=True)
    ev.to_csv(folder/'five_minute_closed_loop.csv',index=False)
    pd.DataFrame({'hour_start_utc':q.index,'DA_commitment_MW':q.to_numpy(),
                  'realized_DA_price_USD_MWh':da.reindex(q.index).to_numpy()}).to_csv(folder/'da_commitments.csv',index=False)
    pd.DataFrame({'hour_start_utc':generated.timestamps_utc,'HL_mean_soc_target_MWhth':hl.reference_mwhth,
                  'RT_DA_spread_historical_proxy_USD_MWh':spread,
                  'HL_deviation_risk_proxy_USD_MWh':risk}).to_csv(folder/'hl_reference.csv',index=False)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
        ti=pd.to_datetime(ev.time_utc,utc=True)
        axs[0].plot(ti,ev.actual_wind_t0_mw,label='Actual wind')
        axs[0].plot(ti,ev.forecast_wind_t0_mw,label='Wind forecast at issue',alpha=.75)
        axs[0].plot(ti,ev.physical_poi_mw,label='Realised POI',alpha=.8)
        axs[0].plot(ti,ev.commanded_discharge_mw,label='Commanded PTES discharge',alpha=.6)
        axs[0].plot(ti,ev.discharge_mw,label='Actual PTES discharge',alpha=.6)
        axs[0].set_ylabel('MW');axs[0].legend(ncol=3,fontsize=8);axs[0].grid(alpha=.2)
        axs[1].plot(ti,ev.charge_mw,label='Executed charge')
        axs[1].plot(ti,ev.discharge_mw,label='Executed discharge')
        axs[1].set_ylabel('MW');axs[1].legend();axs[1].grid(alpha=.2)
        axs[2].plot(ti,ev.tank_mwhth,label='Realized SoC')
        axs[2].plot(ti,ev.ll_target_mwhth,label='LL terminal target',alpha=.7)
        axs[2].set_ylabel('MWhth');axs[2].grid(alpha=.2);axs[2].legend()
        fig.tight_layout();fig.savefig(folder/'dispatch_and_soc.png',dpi=150);plt.close(fig)
    except Exception as ex:
        summary['plot_generation_error']=str(ex)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--day',default='2025-12-18')
    p.add_argument('--hours',type=int,default=2)
    p.add_argument('--scenarios',type=int,default=5)
    p.add_argument('--scenario-model',choices=['tcvae','copula'],default='tcvae')
    p.add_argument('--start-fraction',type=float,choices=[0.,1.],default=0.)
    p.add_argument('--finish-fraction',type=float,choices=[0.,1.],default=None)
    p.add_argument('--hl-backend',choices=['gurobi','highs'],default='gurobi')
    p.add_argument('--ll-backend',choices=['gurobi','highs'],default='gurobi')
    p.add_argument('--hl-limit',type=float,default=600.)
    p.add_argument('--ll-limit',type=float,default=30.)
    p.add_argument('--hl-gap',type=float,default=.01)
    p.add_argument('--ll-gap',type=float,default=.01)
    p.add_argument('--threads',type=int,default=0,help='Gurobi threads (0=automatic)')
    p.add_argument('--solver-output',action='store_true',help='print Gurobi HL log')
    p.add_argument('--out',default=None)
    a=p.parse_args()
    run(day=a.day,hours=a.hours,ns=a.scenarios,model=a.scenario_model,
        start_fraction=a.start_fraction,finish_fraction=a.finish_fraction,
        hl_time_limit=a.hl_limit,ll_time_limit=a.ll_limit,hl_gap=a.hl_gap,ll_gap=a.ll_gap,
        backend=a.ll_backend,hl_backend=a.hl_backend,gurobi_threads=a.threads,
        gurobi_output=a.solver_output,out=a.out)
