#!/usr/bin/env python3
"""Continuous 2-7 day DEVELOPMENT-ONLY stochastic HL / causal five-minute LL replay.

One initial HL day-ahead bid is issued before episode start with a DECLARED
assumed delivery SoC. Subsequent next-day bids are generated at D-1 09:30
America/Chicago from the observed PTES state plus either a causal forward
rollout of an approximate LL policy (default) or the previous HL reference
change (legacy bridge).
Never read future realised wind/RT into any controller input.
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd

from run_development_day import (HERE, ROOT, FORECAST, CHICAGO, load_raw,
                                 clearing_asof, available_da, soc_lookup)
from ptes_mpc.hl_market import solve_hl_market, estimate_spread_from_past
from ptes_mpc.physics import PlantParams,PlantState
from ptes_mpc.harness import simulate_5min
from ptes_mpc.adapters import FrozenProviders
from ptes_mpc.settlement import SettlementLedger
from soc_handover import predict_delivery_soc
from soc_rollout import forecast_rollout, oracle_rollout


def local_midnight(day):
    return pd.Timestamp(day).tz_localize(CHICAGO).tz_convert('UTC')


def validate_days(first_day, days):
    if not (2 <= days <= 7):
        raise ValueError('Choose between 2 and 7 consecutive development days')
    first=pd.Timestamp(first_day)
    if first.tz is not None or not (pd.Timestamp('2025-12-08') <= first <= pd.Timestamp('2025-12-23')):
        raise ValueError('Only safe December 8–23 2025 engineering start dates are supported')
    dates=[(first+pd.DateOffset(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    for day in dates:
        if local_midnight((pd.Timestamp(day)+pd.DateOffset(days=1)).strftime('%Y-%m-%d'))-local_midnight(day)!=pd.Timedelta(hours=24):
            raise ValueError('DST transition not supported by fixed 24h HL market formulation')
    return dates


def _force_spyder_inline_for_display():
    """Restore inline after forecasting imports may have reset backend to Agg.

    In particular, optional analysis modules in the bundled forecasting code
    call matplotlib.use('Agg') after the early Spyder launcher magic. Enabling
    the backend *before* these imports is insufficient.
    """
    from IPython import get_ipython
    shell = get_ipython()
    if shell is not None:
        shell.run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.switch_backend('module://matplotlib_inline.backend_inline')
    backend = str(plt.get_backend())
    if 'inline' not in backend.lower():
        raise RuntimeError(f'Cannot display plots: matplotlib backend is {backend!r}')
    print('Plot display backend:', backend, flush=True)


def make_plot_files(folder, ev, hl_forecasts, commitments, daily_cashflows,
                    daily_hl_runs=None, display_plots=False):
    """Save figures and optionally publish them to Spyder's Plots pane.

    Never force matplotlib's 'Agg' backend here. Spyder's inline backend
    must remain active for figures to appear in its Plots pane.
    """
    # The 84h scenario-generator imports may select Agg after the launcher
    # enabled inline. Restore the Spyder backend at the last responsible moment.
    if display_plots:
        _force_spyder_inline_for_display()
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.grid':True,'grid.alpha':0.25,'figure.dpi':110})
    t=pd.to_datetime(ev['time_utc'],utc=True)
    f=pd.DataFrame(hl_forecasts)
    f['hour_start_utc']=pd.to_datetime(f.hour_start_utc,utc=True)
    q=pd.DataFrame(commitments)
    q['hour_start_utc']=pd.to_datetime(q.hour_start_utc,utc=True)
    dpi=160
    def finish(fig, filename):
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=dpi, bbox_inches='tight')
        if display_plots:
            # Spyder Inline: publishes this figure to the Plots pane.
            # No plt.close() before display: that was the old bug.
            plt.show()
            # Inline may close the figure automatically. Ensure no figure
            # remains open between six multi-day charts.
            plt.close(fig)
        else:
            plt.close(fig)
    # Dedicated plots are more readable across a multi-day run.
    fig,ax=plt.subplots(figsize=(14,4.5))
    ax.plot(t,ev.tank_mwhth,label='Realised PTES SoC (thermal MWh)',linewidth=1.5)
    ref=f.drop_duplicates('hour_start_utc').sort_values('hour_start_utc')
    # A newly issued HL plan can have a different predicted delivery-midnight SoC.
    # Plot each plan in its own disconnected segment, not one fictitious continuous
    # trajectory connecting separate forecasts at the daily boundary.
    for i,(delivery_day,chunk) in enumerate(ref.groupby('delivery_day_local',sort=True)):
        chunk=chunk.sort_values('hour_start_utc')
        hours=chunk['hour_start_utc'].tolist()
        values=chunk['hl_reference_mwhth'].tolist()
        if daily_hl_runs is not None:
            matching=[r for r in daily_hl_runs if r['delivery_day_local']==delivery_day]
            if len(matching)==1:
                hours=[local_midnight(delivery_day)]+hours
                values=[float(matching[0]['hl_initial_soc_mwhth'])]+values
        ax.plot(hours,values,label='HL reference, reissued daily' if i==0 else None,
                color='tab:orange',linestyle='--',linewidth=1.5)
    ax.set(ylabel='MWhth',title='Continuous multi-day PTES inventory')
    ax.legend(loc='best');finish(fig,'01_soc_evolution.png')

    fig,ax=plt.subplots(figsize=(14,4.5))
    ax.plot(t,ev.charge_mw,label='Executed charge (MW)',color='tab:blue')
    ax.plot(t,ev.discharge_mw,label='Executed discharge (MW)',color='tab:orange')
    ax.plot(t,ev.commanded_discharge_mw,label='Requested discharge (MW)',alpha=.5,linestyle=':')
    ax.set(ylabel='MW',title='PTES five-minute charge/discharge and regulatory backdown')
    ax.legend(ncol=3);finish(fig,'02_ptes_dispatch.png')

    fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True)
    axs[0].plot(t,ev.actual_wind_t0_mw,label='Realised wind 5-min',linewidth=1.4)
    axs[0].plot(t,ev.forecast_wind_t0_mw,label='XGBoost wind forecast issued at interval start',alpha=.72,linewidth=1)
    axs[0].set(ylabel='Wind MW',title='Wind: causal five-minute forecast versus actual')
    axs[0].legend()
    axs[1].plot(t,ev.physical_poi_mw,label='Realised physical POI',linewidth=1.4)
    axs[1].plot(t,ev.forecast_poi_t0_mw,label='Forecast POI from planned action',alpha=.72)
    forbid=q[['hour_start_utc','DA_commitment_MW']].set_index('hour_start_utc')['DA_commitment_MW']
    axs[1].step(forbid.index,forbid.values,where='post',label='Accepted DA commitment',color='tab:purple',alpha=.8)
    axs[1].set(ylabel='MW',title='Physical POI forecast, actual and DA commercial position')
    axs[1].legend(ncol=3);finish(fig,'03_wind_and_poi_forecast_vs_actual.png')

    fig,axs=plt.subplots(2,1,figsize=(14,7),sharex=True)
    axs[0].plot(t,ev.actual_wind_t0_mw,label='Available measured wind (MW)')
    axs[0].plot(t,ev.dispatched_wind_mw,label='Delivered wind after curtailment (MW)')
    axs[0].set(ylabel='MW',title='Measured wind is unchanged; wind dispatch responds to control')
    axs[0].legend()
    axs[1].plot(t,ev.wind_dispatch_cap_mw,label='MPC farm output cap')
    axs[1].plot(t,ev.curtailed_wind_mw,label='Effective curtailed MW',alpha=.7)
    axs[1].set(ylabel='MW',title='Wind output cap; dispatched ramp ±40 MW/5min, weather-limited downward exceptions')
    axs[1].legend();finish(fig,'07_wind_curtailment.png')

    fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True)
    axs[0].plot(t,ev.actual_rt15_usd_mwh,label='Realised RT15 SPP',linewidth=1.3)
    axs[0].plot(t,ev.forecast_rt15_usd_mwh,label='RT price forecast available at issue',alpha=.78)
    axs[0].set(ylabel='$/MWh',title='Real-time price: four-hour model, first applicable bucket vs realised SPP')
    axs[0].legend()
    fday=f[f.hour_start_utc.isin(q.hour_start_utc)].drop_duplicates('hour_start_utc').set_index('hour_start_utc').sort_index()
    axs[1].plot(fday.index,fday.hl_expected_da_usd_mwh,label='HL expected DA price, issued D−1 09:30')
    axs[1].plot(q.hour_start_utc,q.realized_DA_price_USD_MWh,label='Actual DA clearing price')
    axs[1].set(ylabel='$/MWh',title='Day-ahead predicted price versus actual cleared price')
    axs[1].legend()
    finish(fig,'04_price_forecast_vs_actual.png')

    fig,axs=plt.subplots(2,1,figsize=(14,8),sharex=True)
    axs[0].plot(fday.index,fday.hl_expected_wind_mw,label='HL scenario-weighted wind expectation')
    actual_hour=ev.assign(hour=t.dt.floor('h')).groupby('hour')['actual_wind_t0_mw'].mean()
    axs[0].plot(actual_hour.index,actual_hour.values,label='Realised hourly wind')
    axs[0].set(ylabel='MW',title='Strategic HL wind forecast versus realised power')
    axs[0].legend()
    axs[1].step(q.hour_start_utc,q.DA_commitment_MW,where='post',label='One common HL DA commitment per hour')
    physical_hour=ev.assign(hour=t.dt.floor('h')).groupby('hour')['physical_poi_mw'].mean()
    axs[1].plot(physical_hour.index,physical_hour.values,label='Realised hourly net POI')
    axs[1].set(ylabel='MW',title='Day-ahead bids versus actual energy delivery')
    axs[1].legend();finish(fig,'05_hl_forecasts_and_commitments.png')

    if len(daily_cashflows):
        cash=pd.DataFrame(daily_cashflows)
        fig,ax=plt.subplots(figsize=(11,4.8))
        x=np.arange(len(cash));width=.25
        ax.bar(x-width,cash.da_usd,width,label='DA revenue')
        ax.bar(x,cash.rt_usd,width,label='RT deviation revenue')
        ax.bar(x+width,cash.total_usd,width,label='Total market cashflow')
        ax.set_xticks(x,cash.delivery_day_local,rotation=20)
        ax.set(ylabel='USD (signed)',title='Actual settlement by delivery day (no annualisation)')
        ax.legend();finish(fig,'06_daily_da_rt_settlement.png')


def run(*,day='2025-12-08',days=7,ns=50,model='tcvae',start_fraction=0.,finish_fraction=None,
        backend='gurobi',hl_backend='gurobi',out=None,
        hl_time_limit=120.,ll_time_limit=30.,beta=.01,hl_gap=.02,ll_gap=.02,
        tracking_policy="weighted",tracking_weight_usd_per_mwh=300.,
        tracking_budget_mwh=.25,curtailment_enabled=True,
        gurobi_threads=0,gurobi_seed=1,gurobi_output=False,progress_every=48,
        ll_maximum_time_limit_gap=None,ll_warm_start_enabled=False,
        make_plots=True,display_plots=False, detailed_diagnostics=True,
        ll_diagnostic_stride=3, hl_curtailment_enabled=False, hl_risk_scale=1.0,
        plant_params=None, soc_handover='rollout', rollout_policy='tracking_rule',
        rollout_relax_hours=4.0, rollout_anchor='hl_mean'):
    """Choose a causal bid-to-delivery SoC handover: forward rollout or prior-plan bridge.

    The default tracking-rule rollout is an APPROXIMATION of the single-shot LL-MIQP,
    not an additional optimizer nor demonstrated to improve prediction accuracy.
    """
    if not np.isfinite(hl_risk_scale) or hl_risk_scale<0:
        raise ValueError('HL risk scale must be finite and nonnegative')
    if soc_handover not in ('rollout','bridge'):
        raise ValueError('soc_handover must be rollout or bridge')
    if rollout_anchor not in ('hl_mean','trailing_30d'):
        raise ValueError('rollout_anchor must be hl_mean or trailing_30d')
    if not np.isfinite(rollout_relax_hours) or rollout_relax_hours <= 0:
        raise ValueError('rollout_relax_hours must be finite and positive')
    dates=validate_days(day,int(days))
    if not (2<=ns<=50):raise ValueError('scenario count must be 2..50')
    if start_fraction not in (0.,1.) or finish_fraction not in (None,0.,1.):
        raise ValueError('Only agreed SoC cases: start 0/1, finish None/0/1')
    folder=Path(out) if out is not None else HERE/'results'/f'newweek_{dates[0]}_{days}days_{ns}scenarios_{model}'
    folder.mkdir(parents=True,exist_ok=True)
    diag_dir=folder/'diagnostics'
    if detailed_diagnostics:diag_dir.mkdir(parents=True,exist_ok=True)
    started=time.time();prm=plant_params if plant_params is not None else PlantParams()
    hourly,rt,wind=load_raw()
    da=hourly['DA_price_usd_mwh'].dropna()
    start=local_midnight(dates[0]);end=local_midnight((pd.Timestamp(dates[-1])+pd.DateOffset(days=1)).strftime('%Y-%m-%d'))
    steps=int((end-start)/pd.Timedelta(minutes=5))
    expected_wind_idx=pd.date_range(start,periods=steps,freq='5min',tz='UTC')
    real=wind.reindex(expected_wind_idx)
    if real.isna().any():raise ValueError(f'actual wind missing (no interpolation), first {real[real.isna()].index[0]}')
    past=wind.loc[(wind.index>=start-pd.Timedelta(hours=12))&(wind.index<start)].dropna()
    if len(past)<144:raise ValueError('Need at least 12 hours of prior measured wind')
    for h in pd.date_range(start,end-pd.Timedelta(hours=1),freq='h',tz='UTC'):
        if h not in da.index:raise ValueError(f'Missing actual DA price at {h}')
    for k in pd.date_range(start,end-pd.Timedelta(minutes=15),freq='15min',tz='UTC'):
        if k not in rt.index:raise ValueError(f'Missing actual 15-minute SPP at {k}')
    initial=PlantState(prm.capacity_mwhth*start_fraction,'off',prm.off_min_steps)
    sys.path.insert(0,str(FORECAST))
    from horse_creek_forecasts import JointForecastProvider
    provider=JointForecastProvider(model)
    frozen=FrozenProviders(ROOT,da_cleared_at_for=clearing_asof)
    commitments={};plans={};hl_runs=[];hl_rows=[];rollout_inputs={}
    diagnostic_ll_rows=[];scenario_summary_rows=[];scenario_weight_rows=[]

    def build_plan(delivery_day,initial_inventory,origin,actual_at_bid=None):
        generated=provider.generate(delivery_day,n_raw=500,n_keep=ns,seed=3714)
        target_start=local_midnight(delivery_day)
        if generated.timestamps_utc[0]!=target_start:
            raise AssertionError('HL forecast delivery index mismatch')
        issue=pd.Timestamp(generated.cutoff_utc).tz_convert('UTC')
        required_issue=target_start-pd.Timedelta(hours=14,minutes=30)
        if issue!=required_issue:
            raise AssertionError(f'HL bid cutoff {issue} is not D-1 09:30 local ({required_issue})')
        if actual_at_bid is not None and issue!=actual_at_bid['time']:
            raise AssertionError('HL scenario cutoff does not equal live bid time')
        handover_fields={}
        if callable(initial_inventory):
            initial_inventory,handover_fields=initial_inventory(generated)
        if not np.isfinite(initial_inventory) or not 0 <= initial_inventory <= prm.capacity_mwhth:
            raise ValueError('predicted delivery SoC outside physical storage range')
        if detailed_diagnostics:
            # Pre-optimisation forecast diagnostics: raw model samples and
            # actual selected reduced scenarios; NO future actual observations.
            np.savez_compressed(diag_dir/f'HL_{delivery_day}_raw_and_reduced.npz',
                raw_wind_mw=generated.raw_wind_mw,
                raw_DA_price_usd_mwh=generated.raw_da_price_usd_mwh,
                reduced_wind_mw=generated.wind_mw,
                reduced_DA_price_usd_mwh=generated.da_price_usd_mwh,
                retained_probabilities=generated.weights,
                hour_start_utc=generated.timestamps_utc.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy())
            for scenario_id, probability in enumerate(generated.weights):
                scenario_weight_rows.append(dict(delivery_day_local=delivery_day,
                    scenario_id=scenario_id, weight=float(probability),
                    mean_wind_delivery_mw=float(generated.wind_mw[scenario_id,:24].mean()),
                    mean_DA_price_delivery_usd_mwh=float(generated.da_price_usd_mwh[scenario_id,:24].mean())))
            raw_w,red_w=generated.raw_wind_mw,generated.wind_mw
            raw_p,red_p=generated.raw_da_price_usd_mwh,generated.da_price_usd_mwh
            for hi,hour in enumerate(generated.timestamps_utc):
                scenario_summary_rows.append(dict(delivery_day_local=delivery_day,
                    forecast_issue_utc=issue, hour_start_utc=hour,
                    raw_wind_mean_mw=float(raw_w[:,hi].mean()),
                    retained_weighted_wind_mean_mw=float(np.dot(generated.weights,red_w[:,hi])),
                    raw_wind_q10_mw=float(np.quantile(raw_w[:,hi],0.1)),
                    raw_wind_q90_mw=float(np.quantile(raw_w[:,hi],0.9)),
                    retained_wind_q10_unweighted_mw=float(np.quantile(red_w[:,hi],0.1)),
                    retained_wind_q90_unweighted_mw=float(np.quantile(red_w[:,hi],0.9)),
                    raw_DA_mean_usd_mwh=float(raw_p[:,hi].mean()),
                    retained_weighted_DA_mean_usd_mwh=float(np.dot(generated.weights,red_p[:,hi])),
                    raw_DA_q10_usd_mwh=float(np.quantile(raw_p[:,hi],0.1)),
                    raw_DA_q90_usd_mwh=float(np.quantile(raw_p[:,hi],0.9))))
            (diag_dir/f'HL_{delivery_day}_forecast_metadata.json').write_text(
                json.dumps(generated.metadata,indent=2,default=str),encoding='utf-8')
        spread,risk=estimate_spread_from_past(rt15=rt,da_hourly=da,
                decision_utc=issue,delivery_times_utc=generated.timestamps_utc)
        result=solve_hl_market(generated.wind_mw,generated.da_price_usd_mwh,
            generated.weights,float(initial_inventory),spread,risk,
            time_limit_s=hl_time_limit,mip_rel_gap=hl_gap,solver_backend=hl_backend,
            gurobi_threads=gurobi_threads,gurobi_seed=gurobi_seed,gurobi_output=gurobi_output,
            hl_curtailment_enabled=hl_curtailment_enabled,hl_risk_scale=hl_risk_scale,prm=prm)
        if result.commitment_mw is None or not result.success:
            raise RuntimeError(f'HL failed for {delivery_day}: {result.message}; gap={result.mip_gap}')
        q_times=pd.date_range(target_start,periods=24,freq='h',tz='UTC')
        for h,qval in zip(q_times,result.commitment_mw):
            if h in commitments:raise AssertionError('duplicate DA commitment')
            commitments[h]=float(qval)
        plans[delivery_day]={'start':target_start,'lookup':soc_lookup(target_start,initial_inventory,result.reference_mwhth),
                'initial':float(initial_inventory),'result':result,'issue':issue}
        # Runtime verification on EVERY new plan: prevent another silent
        # datetime-unit regression or accidentally flattened reference.
        _plan = plans[delivery_day]
        _ref = np.r_[float(initial_inventory), np.asarray(result.reference_mwhth, dtype=float)]
        for _hour in (0, 1, 10, 24, 48, 84):
            _actual = _plan['lookup'](target_start + pd.Timedelta(hours=_hour))
            _expected = float(_ref[_hour])
            if not np.isclose(_actual, _expected, atol=1e-6, rtol=1e-10):
                raise AssertionError(
                    f'HL SoC lookup defective at h={_hour}: got {_actual:.6f}, expected {_expected:.6f} '
                    '(check pandas datetime resolution and the trajectory time origin)')
        prior=actual_at_bid or {}
        hl_runs.append({'delivery_day_local':delivery_day,'hl_issued_utc':issue.isoformat(),
             'hl_initial_soc_mwhth':float(initial_inventory),'hl_initial_soc_source':origin,
             'soc_observed_at_bid_mwhth':prior.get('observed_soc_mwhth'),
             'soc_bridge_reference_change_mwhth':prior.get('bridge_change_mwhth'),
             'old_hl_reference_at_bid_mwhth':prior.get('old_ref_at_bid_mwhth'),
             'old_hl_reference_at_delivery_mwhth':prior.get('old_ref_at_delivery_mwhth'),
             'soc_prediction_before_bounds_mwhth':prior.get('predicted_before_bounds_mwhth'),
             'soc_bridge_clip_correction_mwhth':prior.get('clip_correction_mwhth'),
             'soc_bridge_method':handover_fields.get('soc_handover_method','explicit_previous_plan_reference_delta'),
             **{k:v for k,v in handover_fields.items() if k!='soc_handover_method'},
             'solver_backend':hl_backend,'solver_runtime_seconds':result.runtime_seconds,
             'solver_gap':result.mip_gap,'solver_success':result.success,
             'scenarios':ns,'mean_wind_day1_mw':float(np.dot(generated.weights,generated.wind_mw[:,:24]).mean()),
             'predicted_DA_total_mwh':float(np.sum(result.commitment_mw)),
             'predicted_DA_min_mw':float(np.min(result.commitment_mw)),
             'hl_curtailment_enabled':bool(hl_curtailment_enabled),
             'hl_risk_scale':float(hl_risk_scale),
             'expected_dispatch_wind_day1_mwh':float(np.sum(result.expected_wind_dispatch_mw[:24]))})
        for h in range(84):
            hour=generated.timestamps_utc[h]
            hl_rows.append({'forecast_issue_utc':issue,'delivery_day_local':delivery_day,
                'hour_start_utc':hour,
                'hl_expected_wind_mw':float(np.dot(generated.weights,generated.wind_mw[:,h])),
                'hl_expected_dispatched_wind_mw':float(result.expected_wind_dispatch_mw[h]),
                'hl_expected_da_usd_mwh':float(np.dot(generated.weights,generated.da_price_usd_mwh[:,h])),
                'hl_reference_mwhth':float(result.reference_mwhth[h]),
                'rt_da_spread_proxy_usd_mwh':float(spread[h]),
                'risk_proxy_usd_mwh':float(risk[h])})
        if detailed_diagnostics:
            pd.DataFrame(scenario_summary_rows).to_csv(diag_dir/'HL_raw_vs_reduced_hourly.csv',index=False)
            pd.DataFrame(scenario_weight_rows).to_csv(diag_dir/'HL_retained_scenario_weights.csv',index=False)
            pd.DataFrame(hl_runs).to_csv(diag_dir/'HL_daily_checkpoint.csv',index=False)
        print(f'HL issued {issue} for {delivery_day}: initial SoC={initial_inventory:.2f} MWhth;'
              f' q range [{min(result.commitment_mw):.2f},{max(result.commitment_mw):.2f}] MW;'
              f' gap={result.mip_gap:.3%}; runtime={result.runtime_seconds if result.runtime_seconds is not None else float("nan"):.1f}s',flush=True)
        return dict(zip(q_times,result.commitment_mw))

    build_plan(dates[0],initial.energy_mwhth,'episode-start assumption: delivery-midnight SoC')
    def reference(t):
        instant=pd.Timestamp(t).tz_convert('UTC')
        selected=max((x for x in plans.values() if x['start']<=instant),
                     key=lambda x:x['start'],default=None)
        if selected is None:selected=plans[dates[0]]
        return selected['lookup'](instant)

    future_issue_times={local_midnight(x)-pd.Timedelta(hours=14,minutes=30):x for x in dates[1:]}
    # On the preceding day at 09:30 LOCAL. Dec has UTC-6, thus 15:30 UTC.
    # Calendar-day subtraction uses a stable fixed 24-hour period on these DST-free dates.
    def pre_step(t,current):
        if t not in future_issue_times:return None
        target_day=future_issue_times[t];delivery_midnight=local_midnight(target_day)
        # Causal bridge: actual state at bidding time plus the old plan's
        # unexecuted reference change until delivery. This is explicitly an
        # approximation and NEVER reads the future actual SoC or future wind.
        # Never resolve the reference through the global active-plan selector
        # at a FUTURE instant. That selector can point to a newly registered
        # delivery-day plan; the bridge must be evaluated against the previous
        # day's immutable bid forecast on BOTH endpoints.
        old_plan=max((plan for plan in plans.values() if plan['start']<=t),
                     key=lambda plan:plan['start'],default=None)
        if old_plan is None or old_plan['start']!=delivery_midnight-pd.Timedelta(hours=24):
            raise AssertionError('bid-to-delivery bridge must use previous delivery-day HL plan')
        bridge=predict_delivery_soc(
            bidding_utc=t,delivery_utc=delivery_midnight,
            observed_soc_mwhth=current.energy_mwhth,
            previous_plan_start_utc=old_plan['start'],
            previous_reference_lookup=old_plan['lookup'],
            capacity_mwhth=prm.capacity_mwhth)
        print(f'HL SoC bridge {target_day}: observed={bridge.observed_at_bid:.2f}, '
              f'old ref@bid={bridge.previous_ref_at_bid:.2f}, '
              f'old ref@delivery={bridge.previous_ref_at_delivery:.2f}, '
              f'raw delta={bridge.planned_change:+.2f}, '
              f'preclip={bridge.predicted_before_bounds:.2f}, '
              f'predicted={bridge.predicted_at_delivery:.2f} MWhth',flush=True)
        prior={'time':t,'observed_soc_mwhth':current.energy_mwhth,
               'bridge_change_mwhth':bridge.planned_change,
               'old_ref_at_bid_mwhth':bridge.previous_ref_at_bid,
               'old_ref_at_delivery_mwhth':bridge.previous_ref_at_delivery,
               'predicted_before_bounds_mwhth':bridge.predicted_before_bounds,
               'clip_correction_mwhth':bridge.clipped_by_bounds}
        if soc_handover=='bridge':
            return build_plan(target_day,bridge.predicted_at_delivery,
                'measured at bid + explicit previous HL remaining reference change',prior)
        # Historic measured values are restricted to before t; future path uses a point forecast.
        state_at_bid=current
        measured=pd.concat([past,real.loc[real.index<t]]).sort_index()
        if measured.empty or measured.index.max()>=t:
            raise AssertionError('SoC rollout measured wind is missing or includes future wind')
        def initial_from_rollout(generated):
            if pd.Timestamp(generated.cutoff_utc).tz_convert('UTC')!=t:
                raise AssertionError('HL scenario cutoff must equal rollout issue time')
            if rollout_anchor=='hl_mean':
                anchor=float(np.dot(generated.weights,generated.wind_mw[:,0]))
                n_anchor_obs=None
            else:
                # Important: do not truncate this history to the LL's 12h seed.
                # Early Dec has fewer than 30d available; log the actual count.
                trailing=wind.loc[(wind.index>=t-pd.Timedelta(days=30))&(wind.index<t)].dropna()
                if trailing.empty:raise ValueError('no pre-bid wind for trailing anchor')
                anchor=float(trailing.mean())
                n_anchor_obs=len(trailing)
            ro=forecast_rollout(issue_utc=t,delivery_utc=delivery_midnight,state=state_at_bid,
                measured_wind_5min=measured,short_term_wind_predict=frozen.wind,
                anchor_wind_mw=anchor,commitments_hourly=commitments,
                da_prices_hourly=available_da(t,da),policy=rollout_policy,
                relax_hours=rollout_relax_hours,prm=prm,
                curtailment_enabled=curtailment_enabled)
            rollout_inputs[target_day]=dict(issue=t,delivery=delivery_midnight,state=state_at_bid)
            if detailed_diagnostics:
                ro.trajectory.to_csv(diag_dir/f'SoC_rollout_{target_day}.csv',index=False)
            print(f'HL SoC rollout {target_day}: observed={state_at_bid.energy_mwhth:.2f} MWhth, '
                  f'wind mean={ro.gap_wind_forecast_mean_mw:.1f} MW, '
                  f'accepted q mean={ro.gap_commitment_mean_mw:.1f} MW, '
                  f'predicted={ro.predicted_soc_mwhth:.2f} MWhth '
                  f'(old bridge={bridge.predicted_at_delivery:.2f})',flush=True)
            return ro.predicted_soc_mwhth,dict(
                soc_handover_method=f'forward_rollout_{ro.policy}',
                bridge_predicted_soc_mwhth=bridge.predicted_at_delivery,
                rollout_relax_hours=rollout_relax_hours,rollout_anchor=rollout_anchor,
                rollout_anchor_wind_observation_count=n_anchor_obs,
                **ro.log_fields())
        return build_plan(target_day,initial_from_rollout,
            'measured at bid + causal forward rollout of LL policy approximation',prior)

    def wind_predictor(t,history):return frozen.wind(t,history)
    def rt_predictor(t,completed_rt,_unused_da,completed_wind):
        return frozen.rt(t,completed_rt,available_da(t,da),completed_wind)

    def capture_ll_horizon(t,state,wind_h,price_h,sol,reference):
        if not detailed_diagnostics:return
        for j,(fw,fp) in enumerate(zip(wind_h,price_h)):
            diagnostic_ll_rows.append(dict(
                issue_utc=t,forecast_target_utc=t+pd.Timedelta(minutes=5*j),
                lead_minutes=5*j, initial_soc_mwhth=float(state.energy_mwhth),
                initial_mode=state.mode, forecast_wind_mw=float(fw),
                forecast_RT_price_usd_mwh=float(fp),
                planned_charge_mw=float(sol.charge_mw[j]) if sol.charge_mw is not None else np.nan,
                planned_discharge_mw=float(sol.discharge_mw[j]) if sol.discharge_mw is not None else np.nan,
                planned_wind_dispatch_cap_mw=(float(sol.wind_cap_mw[j]) if
                    getattr(sol,'wind_cap_mw',None) is not None else np.nan),
                planned_soc_after_mwhth=float(sol.tank_mwhth[j]) if sol.tank_mwhth is not None else np.nan,
                planned_mode=sol.modes[j] if sol.modes is not None else None,
                terminal_reference_mwhth=float(reference),
                solver_gap=float(sol.mip_gap) if sol.mip_gap is not None else np.nan))

    target=None if finish_fraction is None else prm.capacity_mwhth*finish_fraction
    ll_started=time.time()
    result=simulate_5min(start_utc=start,n_steps=steps,historic_wind_5min=past,
        realized_wind_5min=real,realized_rt15_price=rt,
        da_commitments_hourly=pd.Series({k:v for k,v in commitments.items() if k<start+pd.Timedelta(days=1)}),
        da_prices_hourly=da,cleared_at_utc=clearing_asof(start),
        wind_predictor=wind_predictor,rt_predictor=rt_predictor,
        initial_state=initial,soc_target_at=reference,prm=prm,
        episode_terminal_target_mwhth=target,terminal_beta_usd_per_mwhth2=beta,
        tracking_policy=tracking_policy,
        tracking_weight_usd_per_mwh=tracking_weight_usd_per_mwh,
        tracking_budget_mwh=tracking_budget_mwh,curtailment_enabled=curtailment_enabled,
        solver_backend=backend,time_limit_s=ll_time_limit,mip_rel_gap=ll_gap,
        maximum_time_limit_gap=ll_maximum_time_limit_gap,
        warm_start_enabled=ll_warm_start_enabled,
        before_step=pre_step,progress_every=progress_every,
        diagnostic_callback=capture_ll_horizon if detailed_diagnostics else None,
        diagnostic_stride=ll_diagnostic_stride)
    ll_runtime=time.time()-ll_started
    ev=result.events
    if len(ev)!=steps:raise AssertionError('missing LL steps')
    if len(hl_runs)!=days:raise AssertionError('HL must be issued once per delivery day')
    if ev['physical_poi_mw'].abs().max()>prm.poi_limit_mw+1e-5:raise AssertionError('POI violation')
    # Retain exact Gurobi gap transparency: distinguish target-certified solves
    # from bounded, feasible incumbents accepted after the LL time limit.
    if ev['mip_gap'].notna().any():
        bad_gap=ev['mip_gap'].gt(ll_gap+1e-8) & ~ev['ll_accepted_time_limit_incumbent']
        if bad_gap.any():raise AssertionError('LL gap exceeded target without explicit time-limit exception')
        if ll_maximum_time_limit_gap is not None and (
            ev.loc[ev.ll_accepted_time_limit_incumbent,'mip_gap']>
            ll_maximum_time_limit_gap+1e-8).any():
            raise AssertionError('accepted LL incumbent exceeded permitted gap')
    if abs(ev.physical_poi_mw.sum()/12-result.cashflow['phys_mwh'])>1e-6:
        raise AssertionError('ledger physical energy mismatch')
    if abs(result.cashflow['total_usd']-result.cashflow['da_usd']-result.cashflow['rt_usd'])>1e-6:
        raise AssertionError('cashflow identity mismatch')
    if ev.rt_forecast_refreshed_utc.nunique()!=days*96:
        raise AssertionError('RT quarter-hour forecast update count incorrect')
    q=pd.Series(commitments).sort_index()
    if len(q)!=days*24:raise AssertionError('must have one signed bid per delivery hour')
    if q.min()<-prm.charge_max_mw-1e-5:raise AssertionError('commercial purchase exceeds PTES rating')
    qout=pd.DataFrame({'hour_start_utc':q.index,'DA_commitment_MW':q.values,
                       'realized_DA_price_USD_MWh':da.reindex(q.index).to_numpy(float)})
    # Use only the delivery day's issued forecast for comparisons; other 60h
    # buffer forecasts are available in the complete raw output.
    f=pd.DataFrame(hl_rows)
    f['hour_start_utc']=pd.to_datetime(f.hour_start_utc,utc=True)
    issued_day=f[f['hour_start_utc'].dt.tz_convert(CHICAGO).dt.strftime('%Y-%m-%d')==f.delivery_day_local].copy()
    if len(issued_day)!=days*24:raise AssertionError('expected 24 forecast observations per delivery day')
    ledger=SettlementLedger(q.to_dict(),da.to_dict(),rt.to_dict())
    for r in ev.itertuples(index=False):ledger.record(r.time_utc,r.physical_poi_mw)
    daily_cash=[]
    for d in dates:
        h_start=local_midnight(d);h_end=h_start+pd.Timedelta(hours=24)
        hours_idx=pd.date_range(h_start,periods=24,freq='h',tz='UTC')
        quarters_idx=pd.date_range(h_start,periods=96,freq='15min',tz='UTC')
        da_cash=sum(q[h]*da[h] for h in hours_idx)
        rt_cash=sum(ledger.quarter_settlement(k) for k in quarters_idx)
        block=ev[(ev.time_utc>=h_start)&(ev.time_utc<h_end)]
        daily_cash.append({'delivery_day_local':d,'da_usd':float(da_cash),'rt_usd':float(rt_cash),
           'total_usd':float(da_cash+rt_cash),'realized_poi_mwh':float(block.physical_poi_mw.sum()/12),
           'scheduled_da_mwh':float(q.reindex(hours_idx).sum()),
           'charge_mwh':float(block.charge_mw.sum()/12),'discharge_mwh':float(block.discharge_mw.sum()/12),
           'final_soc_mwhth':float(block.tank_mwhth.iloc[-1]),
           'wind_mwh':float(block.actual_wind_t0_mw.sum()/12),
           'curtailed_wind_mwh':float(block.curtailed_wind_mw.sum()/12.),
           'dispatched_wind_mwh':float(block.dispatched_wind_mw.sum()/12.),
           'regulatory_interventions':int(block.safety_override.sum())})
    if abs(sum(x['total_usd'] for x in daily_cash)-result.cashflow['total_usd'])>1e-6:
        raise AssertionError('daily settlements do not reconcile with overall ledger')
    # Compute actual SoC at midnight from preceding interval; first start is assumed.
    for hr in hl_runs:
        midnight=local_midnight(hr['delivery_day_local'])
        observed=initial.energy_mwhth if midnight==start else float(ev.loc[ev.time_utc==midnight-pd.Timedelta(minutes=5),'tank_mwhth'].iloc[0])
        hr['realized_delivery_start_soc_mwhth']=observed
        hr['predicted_minus_realized_delivery_soc_mwhth']=hr['hl_initial_soc_mwhth']-observed
        if hr.get('bridge_predicted_soc_mwhth') is not None:
            hr['bridge_predicted_minus_realized_mwhth']=hr['bridge_predicted_soc_mwhth']-observed
        ri=rollout_inputs.get(hr['delivery_day_local'])
        if ri is not None:
            # POST-HOC only. Never expose realised future wind to the bid or LL.
            oracle=oracle_rollout(issue_utc=ri['issue'],delivery_utc=ri['delivery'],
                state=ri['state'],realised_wind_5min=real,
                commitments_hourly=commitments,da_prices_hourly=da,
                policy=rollout_policy,prm=prm,
                curtailment_enabled=curtailment_enabled)
            hr['POSTHOC_oracle_rollout_soc_mwhth']=oracle.predicted_soc_mwhth
            hr['POSTHOC_realised_gap_wind_mean_mw']=oracle.gap_wind_forecast_mean_mw
            hr['POSTHOC_error_from_gap_wind_forecast_mwhth']=hr['rollout_predicted_soc_mwhth']-oracle.predicted_soc_mwhth
            hr['POSTHOC_error_from_policy_model_mwhth']=oracle.predicted_soc_mwhth-observed
    ev.to_csv(folder/'five_minute_closed_loop.csv',index=False)
    qout.to_csv(folder/'da_commitments.csv',index=False)
    f.to_csv(folder/'hl_hourly_all_forecast_horizons.csv',index=False)
    issued_day.to_csv(folder/'hl_issued_delivery_day_forecasts.csv',index=False)
    pd.DataFrame(hl_runs).to_csv(folder/'hl_daily_solver_and_soc_handover.csv',index=False)
    pd.DataFrame(daily_cash).to_csv(folder/'daily_cashflow.csv',index=False)
    forecast_metrics=[]
    for d in dates:
        lo=local_midnight(d);hi=lo+pd.Timedelta(hours=24)
        v=ev[(ev.time_utc>=lo)&(ev.time_utc<hi)]
        fd=issued_day[issued_day.delivery_day_local==d].set_index('hour_start_utc')
        actual_hourly_wind=v.assign(hour=v.time_utc.dt.floor('h')).groupby('hour')['actual_wind_t0_mw'].mean()
        actual_da=da.reindex(fd.index)
        forecast_metrics.append({'delivery_day_local':d,
            'wind_5min_first_step_mae_mw':float((v.forecast_wind_t0_mw-v.actual_wind_t0_mw).abs().mean()),
            'wind_5min_first_step_bias_mw':float((v.forecast_wind_t0_mw-v.actual_wind_t0_mw).mean()),
            'hl_wind_next_day_mae_mw':float((fd.hl_expected_wind_mw-actual_hourly_wind).abs().mean()),
            'hl_da_price_next_day_mae_usd_mwh':float((fd.hl_expected_da_usd_mwh-actual_da).abs().mean()),
            'rt15_first_bucket_mae_usd_mwh':float((v.forecast_rt15_usd_mwh-v.actual_rt15_usd_mwh).abs().mean()),
            'forecast_poi_first_step_mae_mw':float((v.forecast_poi_t0_mw-v.physical_poi_mw).abs().mean())})
    pd.DataFrame(forecast_metrics).to_csv(folder/'forecast_accuracy_by_day.csv',index=False)
    if detailed_diagnostics:
        pd.DataFrame(diagnostic_ll_rows).to_csv(diag_dir/'LL_prediction_horizons_every_15min.csv.gz',
                                                index=False,compression='gzip')
        pd.DataFrame(scenario_summary_rows).to_csv(diag_dir/'HL_raw_vs_reduced_hourly.csv',index=False)
        pd.DataFrame(scenario_weight_rows).to_csv(diag_dir/'HL_retained_scenario_weights.csv',index=False)
        qindex=qout.set_index('hour_start_utc').sort_index()
        hh=ev.assign(hour_start_utc=ev.time_utc.dt.floor('h'))
        agg=hh.groupby('hour_start_utc').agg(
            actual_wind_mw=('actual_wind_t0_mw','mean'),
            physical_POI_mw=('physical_poi_mw','mean'),
            charging_mwh=('charge_mw',lambda x:float(x.sum()/12)),
            discharging_mwh=('discharge_mw',lambda x:float(x.sum()/12)),
            standby_count=('actual_mode',lambda x:int((x=='standby').sum())),
            regulatory_interventions=('safety_override','sum'),
            end_soc_mwhth=('tank_mwhth','last'))
        hourly_audit=qindex.join(agg).join(
            issued_day.set_index('hour_start_utc')[['hl_expected_wind_mw','hl_expected_da_usd_mwh',
                'rt_da_spread_proxy_usd_mwh','risk_proxy_usd_mwh']],how='left')
        hourly_audit['DA_revenue_usd']=hourly_audit.DA_commitment_MW*hourly_audit.realized_DA_price_USD_MWh
        hourly_rt=[];mean_rt=[];neg_rt=[]
        for hour,row in hourly_audit.iterrows():
            quarter_idx=pd.date_range(hour,periods=4,freq='15min',tz='UTC')
            q_rev=np.array([ledger.quarter_settlement(k) for k in quarter_idx],float)
            rp=np.array([rt[k] for k in quarter_idx],float)
            hourly_rt.append(float(q_rev.sum()))
            mean_rt.append(float(rp.mean()))
            neg_rt.append(int((rp<0).sum()))
        hourly_audit['RT_revenue_usd']=hourly_rt
        hourly_audit['RT_average_SPP_usd_mwh']=mean_rt
        hourly_audit['RT_negative_quarters_count']=neg_rt
        hourly_audit['net_deviation_mwh']=hourly_audit.physical_POI_mw-hourly_audit.DA_commitment_MW
        hourly_audit['total_settlement_usd']=hourly_audit.DA_revenue_usd+hourly_audit.RT_revenue_usd
        hourly_audit.to_csv(diag_dir/'hourly_commitment_physical_and_settlement_audit.csv')
        # Diagnostic wind-only settlement benchmarks: all RT, and a causal
        # D-2 hourly persistence commitment for D (known at D-1 09:30).
        benchmarks=[]
        for d in dates:
            h_idx=pd.date_range(local_midnight(d),periods=24,freq='h',tz='UTC')
            h_rt=rt.loc[(rt.index>=h_idx[0])&(rt.index<h_idx[-1]+pd.Timedelta(hours=1))]
            actual_wind_hour=ev.assign(hour=ev.time_utc.dt.floor('h')).groupby('hour').actual_wind_t0_mw.mean().reindex(h_idx)
            baseline_q_zero=np.zeros(24)
            lag_hours=[h-pd.Timedelta(hours=48) for h in h_idx]
            hist_lag=wind.groupby(wind.index.floor('h')).mean().reindex(lag_hours).to_numpy(float)
            if not np.isfinite(hist_lag).all():
                raise ValueError(f'48h-lag wind-only comparator unavailable on {d}')
            for name, q_b in [('wind_only_RT_only_zero_DA',baseline_q_zero),
                               ('wind_only_causal_Dminus2_wind_DA',np.clip(hist_lag,0.,prm.poi_limit_mw))]:
                q_map=dict(zip(h_idx,q_b)); cash_da=float(np.dot(q_b,da.reindex(h_idx)))
                cash_rt=0.
                for k,price in h_rt.items():
                    interval_wind=real.loc[k:k+pd.Timedelta(minutes=10)]
                    if len(interval_wind)!=3:raise AssertionError('incomplete wind quarter in benchmark')
                    energy=float(interval_wind.sum()/12)
                    cash_rt+=float(price)*(energy-q_map[k.floor('h')]/4)
                benchmarks.append(dict(delivery_day_local=d,benchmark=name,
                    DA_cashflow_usd=cash_da,RT_cashflow_usd=cash_rt,
                    total_cashflow_usd=cash_da+cash_rt,
                    wind_generation_mwh=float(actual_wind_hour.sum())))
        bench_df=pd.DataFrame(benchmarks)
        bench_df.to_csv(diag_dir/'wind_only_benchmark_settlement.csv',index=False)
        for item in daily_cash:
            d=item['delivery_day_local']
            rtb=bench_df[(bench_df.delivery_day_local==d)&(bench_df.benchmark=='wind_only_RT_only_zero_DA')]
            lagb=bench_df[(bench_df.delivery_day_local==d)&(bench_df.benchmark=='wind_only_causal_Dminus2_wind_DA')]
            item['wind_only_RT_only_usd']=float(rtb.total_cashflow_usd.iloc[0])
            item['wind_only_Dminus2_DA_usd']=float(lagb.total_cashflow_usd.iloc[0])
            item['PTES_minus_wind_only_RT_only_usd']=item['total_usd']-item['wind_only_RT_only_usd']
            item['PTES_minus_wind_only_Dminus2_DA_usd']=item['total_usd']-item['wind_only_Dminus2_DA_usd']
        pd.DataFrame(daily_cash).to_csv(folder/'daily_cashflow.csv',index=False)
        diagnostic_manifest=dict(delivery_dates=dates,raw_scenarios=500,retained_scenarios=ns,
            notes=['2025 Dec wind is within training data: NOT independent OOS.',
                   'All wind-only baselines use exactly the same physical wind record and realised prices.',
                   'Wind-only D-2 commitment is chosen using lagged historical wind available before D-1 09:30.',
                   'Wind-only RT-only (DA=0) is a diagnostic settlement baseline, not necessarily good commercial bidding.',
                   'Negative settlement cashflow does not by itself imply negative incremental PTES contribution.',
                   'Historical spread and risk are provisional HL bidding assumptions, not observed forward RT bids.',
                   'Raw/reduced HL forecast archives and LL horizon snapshots never contain future actual measurements at issue time.'],
            diagnostics=['HL_raw_vs_reduced_hourly.csv','HL_retained_scenario_weights.csv',
                'HL_YYYY-MM-DD_raw_and_reduced.npz','LL_prediction_horizons_every_15min.csv.gz',
                'hourly_commitment_physical_and_settlement_audit.csv',
                'wind_only_benchmark_settlement.csv'])
        (diag_dir/'DIAGNOSTICS_MANIFEST.json').write_text(json.dumps(diagnostic_manifest,indent=2))

    if make_plots:
        make_plot_files(folder,ev,issued_day,qout,daily_cash,hl_runs,
                        display_plots=display_plots)
    summary={'status':'completed engineering development run; NOT an independent out-of-sample economic result',
        'start_local':dates[0],'days':days,'end_exclusive_local':str(pd.Timestamp(dates[-1])+pd.DateOffset(days=1))[:10],
        'note':'Dec 2025 wind is in frozen XGBoost training. HL RT–DA spread remains a provisional financial proxy.',
        'diagnostics_enabled':bool(detailed_diagnostics),
        'initial_soc_mwhth':initial.energy_mwhth,'final_soc_mwhth':result.final_state.energy_mwhth,
        'soft_episode_target_mwhth':target,'soft_terminal_penalty_usd_not_cashflow':result.terminal_soft_penalty_usd,
        'handover_fixed_version':('explicit_prior_plan_reference_delta_v1' if soc_handover=='bridge'
                                  else 'forward_rollout_ramp40_v2'),
        'soc_handover_method':soc_handover,
        'soc_rollout_settings':(None if soc_handover=='bridge' else dict(
            policy=rollout_policy if isinstance(rollout_policy,str) else getattr(rollout_policy,'__name__','custom'),
            relax_hours=rollout_relax_hours,anchor=rollout_anchor,
            curtailment_enabled=curtailment_enabled,
            forecast='frozen LL 5min XGBoost first 4h then relaxation')),
        'num_hl_runs':len(hl_runs),'hl_scenarios':ns,'hl_backend':hl_backend,
        'hl_wind_curtailment_enabled':bool(hl_curtailment_enabled),
        'hl_risk_scale':float(hl_risk_scale),
        'hl_worst_gap':float(max(x['solver_gap'] for x in hl_runs)),
        'hl_daily_solver_results':hl_runs,
        'll_tracking_policy':tracking_policy,'ll_tracking_weight_usd_per_mwh':tracking_weight_usd_per_mwh,
        'total_electricity_revenue_USD':float(result.cashflow['total_usd']),
        'DA_USD':float(result.cashflow['da_usd']),
        'RT_USD':float(result.cashflow['rt_usd']),
        'DA_sold_mwh':float(q.sum()),
        'available_wind_mwh':float(ev.actual_wind_t0_mw.sum()/12.),
        'delivered_wind_mwh':float(ev.dispatched_wind_mw.sum()/12.),
        'final_energy_storage_mwhth':float(result.final_state.energy_mwhth),
        'll_tracking_budget_weighted_mwh':tracking_budget_mwh,
        'll_terminal_soc_beta_usd_per_mwhth2':beta,
        'wind_curtailment_enabled':bool(curtailment_enabled),
        'wind_dispatch_ramp_mw_per_5min':prm.wind_dispatch_ramp_mw_per_5min,
        'curtailed_wind_mwh':result.curtailed_wind_mwh,
        'realized_abs_da_quarterly_deviation_mwh':result.abs_da_deviation_mwh,
        'max_abs_wind_dispatch_change_mw_per_5min':result.max_abs_dispatched_wind_ramp_mw_per_5min,
        'positive_wind_ramp_violations':result.positive_wind_ramp_violations,
        'unavoidable_down_wind_ramp_exceptions':result.unavoidable_down_wind_ramp_exceptions,
        'num_ll_solves':len(ev),'ll_backend':backend,'ll_worst_gap':float(ev.mip_gap.max()),
        'll_solver_failures':result.solve_failures,
        'll_time_limit_incumbents_accepted':result.accepted_time_limit_incumbents,
        'maximum_observed_ll_mip_gap':result.max_observed_ll_gap,'regulatory_interventions':result.regulatory_interventions,
        'max_abs_ptes_change_mw_per_5min':result.max_executed_ptes_power_ramp_mw_per_5min,
        'total_cashflow_usd':result.cashflow,
        'daily_cashflows_and_benchmarks':daily_cash,
        'financial_benchmarks':({
            'wind_only_RT_only_usd':float(sum(x['wind_only_RT_only_usd'] for x in daily_cash)),
            'wind_only_Dminus2_DA_usd':float(sum(x['wind_only_Dminus2_DA_usd'] for x in daily_cash)),
            'PTES_minus_RT_only_wind_usd':float(sum(x['PTES_minus_wind_only_RT_only_usd'] for x in daily_cash)),
            'PTES_minus_Dminus2_DA_wind_usd':float(sum(x['PTES_minus_wind_only_Dminus2_DA_usd'] for x in daily_cash))}
            if detailed_diagnostics else None),
        'hl_soc_bridge':('measured SoC at D-1 09:30 plus previous HL reference change, causal heuristic' if soc_handover=='bridge' else
            'measured state propagated to midnight using causal forecast wind, accepted commitments, plant and simplified LL policy; posthoc oracle uses future data ONLY after episode'),
        'initial_hl_soc_assumption':'first delivery midnight SoC is explicitly supplied by the start case',
        'da_price_availability':'assumed 18:00 local clearance, not validated per-publication timestamps',
        'rt_price_availability':'quarterly RT price is assumed available immediately after interval closes, not independently timestamp validated',
        'total_runtime_seconds':round(time.time()-started,2),'ll_runtime_seconds':round(ll_runtime,2),
        'plot_files':['01_soc_evolution.png','02_ptes_dispatch.png','03_wind_and_poi_forecast_vs_actual.png',
                '04_price_forecast_vs_actual.png','05_hl_forecasts_and_commitments.png','06_daily_da_rt_settlement.png','07_wind_curtailment.png'] if make_plots else []}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
    print(f'Outputs: {folder.resolve()}',flush=True)
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--day',default='2025-12-08')
    a.add_argument('--days',type=int,default=7)
    a.add_argument('--scenarios',type=int,default=50)
    a.add_argument('--scenario-model',choices=['tcvae','copula'],default='tcvae')
    a.add_argument('--start-fraction',type=float,choices=[0.,1.],default=0.)
    a.add_argument('--finish-fraction',type=float,choices=[0.,1.],default=None)
    a.add_argument('--hl-backend',choices=['gurobi','highs'],default='gurobi')
    a.add_argument('--ll-backend',choices=['gurobi','highs'],default='gurobi')
    a.add_argument('--hl-limit',type=float,default=120.)
    a.add_argument('--ll-limit',type=float,default=30.)
    a.add_argument('--hl-gap',type=float,default=.02)
    a.add_argument('--ll-gap',type=float,default=.02)
    a.add_argument('--solver-output',action='store_true')
    a.add_argument('--tracking-policy',choices=['none','weighted','lexicographic'],default='weighted')
    a.add_argument('--tracking-weight',type=float,default=300.)
    a.add_argument('--tracking-budget-mwh',type=float,default=.25)
    a.add_argument('--beta',type=float,default=.01)
    a.add_argument('--no-curtailment',action='store_true')
    a.add_argument('--no-plots',action='store_true')
    a.add_argument('--no-diagnostics',action='store_true')
    a.add_argument('--ll-diagnostic-stride',type=int,default=3)
    a.add_argument('--soc-handover',choices=['rollout','bridge'],default='rollout')
    a.add_argument('--rollout-relax-hours',type=float,default=4.)
    a.add_argument('--rollout-anchor',choices=['hl_mean','trailing_30d'],default='hl_mean')
    a.add_argument('--out',default=None)
    z=a.parse_args()
    run(day=z.day,days=z.days,ns=z.scenarios,model=z.scenario_model,
        start_fraction=z.start_fraction,finish_fraction=z.finish_fraction,
        backend=z.ll_backend,hl_backend=z.hl_backend,out=z.out,
        hl_time_limit=z.hl_limit,ll_time_limit=z.ll_limit,hl_gap=z.hl_gap,ll_gap=z.ll_gap,
        gurobi_output=z.solver_output,make_plots=not z.no_plots,
        tracking_policy=z.tracking_policy,tracking_weight_usd_per_mwh=z.tracking_weight,
        tracking_budget_mwh=z.tracking_budget_mwh,curtailment_enabled=not z.no_curtailment,
        beta=z.beta,
        soc_handover=z.soc_handover,rollout_relax_hours=z.rollout_relax_hours,
        rollout_anchor=z.rollout_anchor,
        detailed_diagnostics=not z.no_diagnostics, ll_diagnostic_stride=z.ll_diagnostic_stride)
