"""Causal coupled wind-only HL (DA LP) / five-minute LL (wind-cap MILP) replay.

No storage, no turbine operating-mode scheduling, no grid charging, no energy
inventory, and no artificial initial/final SoC. The only physical control is
wind curtailment under a 40 MW/5min commanded-cap ramp. DA positions are issued
at D-1 09:30 America/Chicago and remain fixed after issuance. Realized wind
and RT prices enter the process ONLY after the step action is chosen.
"""
from __future__ import annotations
from pathlib import Path
import json, time, sys
import numpy as np
import pandas as pd
from .utils import (ROOT,FORECAST,FORECAST_HL,CHICAGO,load_raw,local_midnight,
                    validate_days,clearing_asof,available_da)
from .adapters import FrozenProviders
from .settlement import SettlementLedger
from .hl import solve_wind_hl
from .ll import solve_wind_ll,execute_wind_cap


def run(*,day='2025-12-08',days=7,ns=50,model='tcvae',
        backend='gurobi',hl_backend='gurobi',tracking_weight=5.,
        hl_risk_scale=1.,offer_cap_fraction=1.,
        ll_time_limit=60.,ll_gap=.025,hl_time_limit=120.,hl_gap=.02,
        progress_every=144,out=None,make_plots=True,forecast_raw_scenarios=500,
        wind_ramp_mw_per_5min=40.):
    dates=validate_days(day,int(days))
    if not 2<=ns<=50 or not 2<=forecast_raw_scenarios or ns>forecast_raw_scenarios:
        raise ValueError('Expect 2..50 retained scenarios and at least that many raw scenarios')
    if tracking_weight<0:raise ValueError('tracking weight must be >=0')
    if not 0<wind_ramp_mw_per_5min<=230.:raise ValueError('Invalid wind command ramp')
    if model not in ('tcvae','copula'):raise ValueError('forecast model must be tcvae or copula')
    if backend not in ('highs','gurobi') or hl_backend not in ('highs','gurobi'):
        raise ValueError('Solver backend must be gurobi or highs')
    folder=Path(out) if out is not None else ROOT/'results'/f'WIND_ONLY_{dates[0]}_{days}d_alpha{tracking_weight:g}'
    folder.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    hourly,rt,wind=load_raw();da=hourly['DA_price_usd_mwh'].dropna()
    start=local_midnight(dates[0]);end=local_midnight((pd.Timestamp(dates[-1])+pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
    steps=int((end-start)/pd.Timedelta(minutes=5))
    stamps=pd.date_range(start,periods=steps,freq='5min',tz='UTC')
    realized_wind=wind.reindex(stamps)
    if realized_wind.isna().any():raise ValueError('Missing actual wind; never interpolate or fill')
    past=wind.loc[(wind.index>=start-pd.Timedelta(hours=12))&(wind.index<start)].dropna()
    if len(past)!=144 or past.index[-1]!=start-pd.Timedelta(minutes=5):
        raise ValueError('Need full prior 12h actual wind, including t-5')
    for h in pd.date_range(start,end-pd.Timedelta(hours=1),freq='h',tz='UTC'):
        if h not in da.index:raise ValueError(f'Missing DA settlement price {h}')
    for k in pd.date_range(start,end-pd.Timedelta(minutes=15),freq='15min',tz='UTC'):
        if k not in rt.index:raise ValueError(f'Missing RT settlement price {k}')
    sys.path.insert(0,str(FORECAST_HL))
    from horse_creek_forecasts import JointForecastProvider
    provider=JointForecastProvider(model)
    frozen=FrozenProviders(FORECAST,da_cleared_at_for=clearing_asof)
    commitments={};hl_rows=[];hl_forecasts=[];first_issue=None

    def issue_bid(delivery_day,expected_issue=None):
        nonlocal first_issue
        scenario=provider.generate(delivery_day,n_raw=forecast_raw_scenarios,
                                   n_keep=ns,seed=3714)
        delivery_start=local_midnight(delivery_day)
        issue=pd.Timestamp(scenario.cutoff_utc).tz_convert('UTC')
        if scenario.timestamps_utc[0]!=delivery_start:
            raise AssertionError('Forecast delivery time mismatch')
        # Exact as-of D-1 09:30, including the first-day pre-episode issue.
        required_issue=delivery_start-pd.Timedelta(hours=14,minutes=30)
        if issue!=required_issue or (expected_issue is not None and issue!=expected_issue):
            raise AssertionError(f'Wrong issue time {issue}, required {required_issue}')
        from .historical_risk import estimate_spread_from_past
        spread,risk=estimate_spread_from_past(rt15=rt,da_hourly=da,decision_utc=issue,
                                               delivery_times_utc=scenario.timestamps_utc)
        solution=solve_wind_hl(scenario.wind_mw,scenario.da_price_usd_mwh,
                  scenario.weights,spread,risk,solver_backend=hl_backend,
                  time_limit_s=hl_time_limit,mip_rel_gap=hl_gap,
                  risk_scale=hl_risk_scale,offer_cap_fraction=offer_cap_fraction)
        if not solution.success or solution.commitment_mw is None:
            raise RuntimeError(f'Wind-only HL bid for {delivery_day} failed: {solution.message}')
        for h,q in enumerate(solution.commitment_mw):
            htime=delivery_start+pd.Timedelta(hours=h)
            if htime in commitments:raise AssertionError('Duplicate issued DA hour')
            commitments[htime]=float(q)
            hl_forecasts.append(dict(delivery_day_local=delivery_day,
                    forecast_issue_utc=issue,hour_start_utc=htime,
                    predicted_available_wind_mw=float(solution.expected_wind_mw[h]),
                    predicted_wind_dispatch_mw=float(solution.expected_dispatch_mw[h]),
                    expected_DA_price_usd_mwh=float(np.dot(scenario.weights,scenario.da_price_usd_mwh[:,h])),
                    historical_RT_minus_DA_spread_usd_mwh=float(spread[h]),
                    historic_risk_proxy_usd_mwh=float(risk[h]),
                    DA_bid_mw=float(q)))
        hl_rows.append(dict(delivery_day_local=delivery_day,forecast_issue_utc=issue,
             HL_model='wind_only_no_storage',forecast_method=model,
             raw_scenarios=forecast_raw_scenarios,retained_scenarios=ns,
             expected_wind_day_mwh=float(np.sum(solution.expected_wind_mw)),
             DA_bid_total_mwh=float(np.sum(solution.commitment_mw)),
             DA_bid_min_mw=float(np.min(solution.commitment_mw)),
             DA_bid_max_mw=float(np.max(solution.commitment_mw)),
             risk_scale=hl_risk_scale,offer_cap_fraction=offer_cap_fraction,
             objective_proxy_usd=solution.objective_proxy_usd,
             solver_seconds=solution.runtime_seconds,solver_gap=solution.mip_gap,
             solver_nvars=solution.model_nvars,solver_nrows=solution.model_nrows))
        if first_issue is None:first_issue=issue
        print(f'WIND-ONLY HL bid issued {issue} for {delivery_day}: '
              f'q range {min(solution.commitment_mw):.1f}..{max(solution.commitment_mw):.1f} MW; '
              f'{sum(solution.commitment_mw):.1f} MWh DA sold',flush=True)
        return {delivery_start+pd.Timedelta(hours=h):float(q)
                for h,q in enumerate(solution.commitment_mw)}

    q_first=issue_bid(dates[0])
    ledger=SettlementLedger(q_first.copy(),da.to_dict(),rt.to_dict())
    issue_times={local_midnight(d)-pd.Timedelta(hours=14,minutes=30):d for d in dates[1:]}
    history=past.copy();prior_dispatch=float(past.iloc[-1])
    last_rt_forecast=None;refresh_time=None
    events=[];solver_seconds=0.;solver_max=0.;solves=0
    ramp=float(wind_ramp_mw_per_5min);rated=230.
    for i,t in enumerate(stamps):
        if t in issue_times:
            fresh=issue_bid(issue_times[t],t)
            for hour,q in fresh.items():
                if hour<t.floor('h') or hour in ledger.q_da_mw:
                    raise AssertionError('Duplicate or late accepted DA position')
                ledger.q_da_mw[hour]=q
        if t.minute%15==0:
            completed=rt.loc[rt.index+pd.Timedelta(minutes=15)<=t]
            if len(completed) and (completed.index+pd.Timedelta(minutes=15)).max()>t:
                raise AssertionError('RT lookahead violation')
            last_rt_forecast=np.asarray(frozen.rt(t,completed.copy(),available_da(t,da),history.copy()),float)
            if last_rt_forecast.shape!=(16,):raise ValueError('Wrong 16-quarter RT forecast')
            refresh_time=t
        if refresh_time is None or t>=refresh_time+pd.Timedelta(minutes=15):
            raise AssertionError('RT forecast stale')
        wind_hat=np.asarray(frozen.wind(t,history.copy()),float)
        if wind_hat.shape!=(48,):raise ValueError('Wind predictor must return 48 steps')
        horizons=pd.date_range(t,periods=48,freq='5min',tz='UTC')
        priced=dict(zip(pd.date_range(refresh_time,periods=16,freq='15min',tz='UTC'),last_rt_forecast))
        prices=[]
        for future in horizons:
            bucket=future.floor('15min')
            if bucket in priced:prices.append(float(priced[bucket]))
            else:
                # Last quarter occasionally crosses 4h forecast boundary.
                # Use last RT prediction, not future uncleared actual DA data.
                prices.append(float(last_rt_forecast[-1]))
        qarray=np.asarray([ledger.q_da_mw.get(x.floor('h'),np.nan) for x in horizons])
        quarter=t.floor('15min');already=float(ledger.energy_mwh.get(quarter,0.))
        offset=(t.minute%15)//5
        if ledger.count_5min.get(quarter,0)!=offset:
            raise AssertionError('Incomplete or duplicated executed quarter')
        tic=time.perf_counter()
        plan=solve_wind_ll(wind_hat,prices,prior_dispatch,qarray,
                already_delivered_quarter_mwh=already,quarter_offset=int(offset),
                tracking_weight_usd_per_mwh=tracking_weight,
                ramp_mw_per_5min=ramp,solver_backend=backend,time_limit_s=ll_time_limit,mip_rel_gap=ll_gap)
        wall=time.perf_counter()-tic
        if not plan.success or plan.cap_mw is None:
            raise RuntimeError(f'Wind-only LL failed at {t}: {plan.message}; gap={plan.mip_gap}')
        solves+=1;solver_seconds+=wall;solver_max=max(solver_max,wall)
        cap=float(plan.cap_mw[0])
        if history.index.max()>=t:
            raise AssertionError('Current/ future wind accessed before decision')
        # The actual wind and settlement RT are OBSERVED only after the LL action.
        actual=float(realized_wind.loc[t])
        delivered,curtailed,exception=execute_wind_cap(
            prior_dispatched=prior_dispatch,wind_actual=actual,cap=cap,rated_mw=rated,ramp=ramp)
        rt_price_actual=float(rt.loc[quarter]);ledger.record(t,delivered)
        events.append(dict(time_utc=t,delivery_day_local=t.tz_convert(CHICAGO).strftime('%Y-%m-%d'),
            wind_forecast_t0_mw=float(wind_hat[0]),
            available_wind_mw=actual,wind_dispatch_cap_mw=cap,
            dispatched_wind_mw=delivered,curtailed_wind_mw=curtailed,
            physical_poi_mw=delivered,previous_dispatched_wind_mw=prior_dispatch,
            dispatched_wind_ramp_mw_per_5min=delivered-prior_dispatch,
            unavoidable_downward_ramp_exception_mw=exception,
            forecast_rt15_usd_mwh=float(prices[0]),actual_rt15_usd_mwh=rt_price_actual,
            accepted_DA_bid_mw=ledger.q_da_mw.get(t.floor('h'),np.nan),
            LL_tracking_weight_usd_per_mwh=tracking_weight,
            LL_solve_wall_s=wall,LL_MIP_gap=plan.mip_gap,
            forecast_first_step_dispatch_mw=float(plan.predicted_dispatch_mw[0]),
            forecast_tracking_abs_mwh=plan.tracking_forecast_abs_mwh))
        prior_dispatch=delivered
        history.loc[t]=actual
        if progress_every and (i+1)%progress_every==0:
            print(f'Wind-only LL {i+1}/{steps}, time={t}, '
                  f'wind={actual:.1f}, dispatched={delivered:.1f}, curtail={curtailed:.1f} MW',flush=True)
    ev=pd.DataFrame(events)
    q=pd.Series(commitments).sort_index()
    if len(q)!=days*24 or len(hl_rows)!=days:
        raise AssertionError('Missing or duplicated DA bidding days')
    if ev['physical_poi_mw'].min()<-1e-6 or ev['physical_poi_mw'].max()>rated+1e-6:
        raise AssertionError('Physical wind-only bounds broken')
    if (ev['dispatched_wind_ramp_mw_per_5min']>ramp+1e-5).any():
        raise AssertionError('Wind output upward ramp broken')
    if (ev['unavoidable_downward_ramp_exception_mw']>1e-5).any():
        print('INFO: Wind downward ramp exception(s) forced by availability.',flush=True)
    cash=ledger.full_cashflow()
    physical_mwh=float(ev.physical_poi_mw.sum()/12)
    if abs(physical_mwh-cash['phys_mwh'])>1e-6:
        raise AssertionError('Settlement generation/ledger mismatch')
    all_q=pd.DataFrame([dict(hour_start_utc=h,DA_commitment_MW=v,
          realized_DA_price_USD_MWh=float(da[h])) for h,v in q.items()])
    benchmark_cash_da=float(sum(q[h]*float(da[h]) for h in q.index))
    benchmark_rt=0.;quarter_rows=[];abs_deviation=0.
    for k in pd.date_range(start,end-pd.Timedelta(minutes=15),freq='15min',tz='UTC'):
        scheduled=float(q[k.floor('h')])/4
        delivered_quarter=float(ledger.energy_mwh[k]);dev=delivered_quarter-scheduled
        raw_energy=float(realized_wind.loc[[k,k+pd.Timedelta(minutes=5),k+pd.Timedelta(minutes=10)]].sum()/12)
        p=float(rt[k]);benchmark_rt+=p*(raw_energy-scheduled)
        abs_deviation+=abs(dev)
        quarter_rows.append(dict(quarter_start_utc=k,DA_commitment_mwh=scheduled,
             delivered_wind_mwh=delivered_quarter,available_wind_mwh=raw_energy,
             wind_curtailed_mwh=raw_energy-delivered_quarter,
             DA_energy_deviation_mwh=dev,absolute_DA_deviation_mwh=abs(dev),
             RT_SPP_usd_mwh=p,RT_settlement_usd=p*dev))
    daily=[]
    quarters=pd.DataFrame(quarter_rows)
    for d in dates:
        hstart=local_midnight(d);hend=hstart+pd.Timedelta(hours=24)
        hs=pd.date_range(hstart,periods=24,freq='h',tz='UTC')
        mask=(ev.time_utc>=hstart)&(ev.time_utc<hend);v=ev.loc[mask]
        qday=quarters[(quarters.quarter_start_utc>=hstart)&(quarters.quarter_start_utc<hend)]
        da_rev=float(sum(q[h]*float(da[h]) for h in hs))
        rt_rev=float(qday.RT_settlement_usd.sum())
        daily.append(dict(delivery_day_local=d,DA_USD=da_rev,RT_USD=rt_rev,
            total_revenue_USD=da_rev+rt_rev,DA_sold_mwh=float(q.reindex(hs).sum()),
            available_wind_mwh=float(v.available_wind_mw.sum()/12),
            delivered_wind_mwh=float(v.dispatched_wind_mw.sum()/12),
            wind_curtailed_mwh=float(v.curtailed_wind_mw.sum()/12),
            abs_DA_deviation_mwh=float(qday.absolute_DA_deviation_mwh.sum())))
    if abs(sum(x['total_revenue_USD'] for x in daily)-cash['total_usd'])>1e-6:
        raise AssertionError('Daily revenue != weekly settlement')
    # This comparator uses the NEW wind-only HL bids, but never curtails.
    benchmark_unrestricted=benchmark_cash_da+benchmark_rt
    summary=dict(case='coupled_wind_only_HL_DA_and_LL_curtailment',
         no_storage=True,days=days,DA_USD=float(cash['da_usd']),RT_USD=float(cash['rt_usd']),
         total_electricity_revenue_USD=float(cash['total_usd']),
         DA_sold_mwh=float(q.sum()),
         available_wind_mwh=float(realized_wind.sum()/12),
         delivered_wind_mwh=physical_mwh,
         curtailed_wind_mwh=float(ev.curtailed_wind_mw.sum()/12),
         abs_quarter_DA_deviation_mwh=float(abs_deviation),
         final_energy_storage_mwhth=None,
         max_positive_wind_ramp_mw_per_5min=float(ev.dispatched_wind_ramp_mw_per_5min.max()),
         unavoidable_downward_ramp_exception_steps=int((ev.unavoidable_downward_ramp_exception_mw>1e-5).sum()),
         wind_command_ramp_limit_mw_per_5min=float(ramp),
         LL_solver_failures=0,LL_solves=solves,
         LL_total_wall_s=float(solver_seconds),LL_max_wall_s=float(solver_max),
         benchmark_same_DA_bids_uncurtailed_wind_revenue_USD=float(benchmark_unrestricted),
         difference_from_same_bids_uncurtailed_USD=float(cash['total_usd']-benchmark_unrestricted),
         wall_time_seconds=float(time.perf_counter()-began))
    ev.to_csv(folder/'five_minute_wind_only.csv',index=False)
    all_q.to_csv(folder/'da_commitments_wind_only.csv',index=False)
    pd.DataFrame(hl_rows).to_csv(folder/'hl_daily_bids.csv',index=False)
    pd.DataFrame(hl_forecasts).to_csv(folder/'hl_delivery_forecasts.csv',index=False)
    pd.DataFrame(daily).to_csv(folder/'daily_cashflow.csv',index=False)
    quarters.to_csv(folder/'quarterly_settlement.csv',index=False)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    (folder/'protocol.json').write_text(json.dumps(dict(
        scenario='wind_only_zero_storage',issue_policy='D-1 09:30 America/Chicago',
        market_model='ERCOT DA financial position plus RT imbalance at realized SPP',
        physical_controls='wind dispatch cap ±40 MW/5min; realized generation=min(cap,available wind)',
        HL_objective='Expected DA+RT settlement - historical RT-DA IQR deviation-risk planning proxy',
        HL_offer_bound='0<=DA sale<=scenario-weighted expected available wind times offer_cap_fraction',
        HL_lookahead='84h joint forecast generated; first 24h optimized (60h tail unused without storage)',
        LL_objective='Forecast RT electricity revenue minus weighted absolute DA tracking preference',
        tracking_weight=tracking_weight,wind_command_ramp_mw_per_5min=ramp,
        hl_risk_scale=hl_risk_scale,
        forecast_model=model,raw_scenarios=forecast_raw_scenarios,retained_scenarios=ns,
        solver_backend=backend,HL_solver_backend=hl_backend,
        DA_price_publication='Study assumption, not verified archive publication time',
        no_SCED_dispatch_instructions=True,
        evaluation_dates=dates,
        warning='December 2025 test window is within wind-forecast training data; development only, not blind out of sample',
        protected_holdouts_accessed=False),indent=2),encoding='utf-8')
    if make_plots:
        from .plots import make_plots as _plot
        _plot(folder,ev,all_q,pd.DataFrame(hl_forecasts),pd.DataFrame(daily),quarters)
    print('WIND-ONLY SUMMARY\n'+json.dumps(summary,indent=2),flush=True)
    print('Results stored in',str(folder.resolve()),flush=True)
    return summary
