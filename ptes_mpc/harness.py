"""Causal 5min LL execution, physical regulation, and post hoc ERCOT cashflow.

Every actual wind value is accessed only AFTER its interval's MPC action is
chosen. Economic settlement uses actual prices, not forecasts. The whole
simulation episode may have an explicit SOFT end-of-episode SoC target.
"""
from dataclasses import dataclass,replace
import numpy as np
import pandas as pd
from .physics import PlantState,PlantParams,execute_step
from .ll_mpc import solve_ll
from .settlement import SettlementLedger

@dataclass
class RunResult:
    events: pd.DataFrame
    cashflow: dict
    final_state: PlantState
    solve_failures: int
    initial_energy_mwhth: float = 0.
    requested_final_energy_mwhth: float|None = None
    actual_terminal_error_mwhth: float|None = None
    terminal_soft_penalty_usd: float = 0.
    regulatory_interventions: int = 0
    forced_off_interventions: int = 0
    clipped_discharge_mwh: float = 0.
    max_executed_ptes_power_ramp_mw_per_5min: float = 0.
    curtailed_wind_mwh: float = 0.
    max_abs_dispatched_wind_ramp_mw_per_5min: float = 0.
    abs_da_deviation_mwh: float = 0.
    positive_wind_ramp_violations: int = 0
    unavoidable_down_wind_ramp_exceptions: int = 0
    max_positive_wind_ramp_mw: float = 0.
    accepted_time_limit_incumbents: int = 0
    max_observed_ll_gap: float = 0.

def simulate_5min(*, start_utc, n_steps, historic_wind_5min, realized_wind_5min,
                  realized_rt15_price, da_commitments_hourly, da_prices_hourly,
                  cleared_at_utc, wind_predictor, rt_predictor,
                  initial_state:PlantState, soc_target_at,
                  prm:PlantParams=PlantParams(),
                  episode_terminal_target_mwhth:float|None=None,
                  terminal_beta_usd_per_mwhth2:float=1.,
                  solver_backend='auto',quadratic_tangents=401,
                  time_limit_s=20.,mip_rel_gap=.02,stop_on_solver_failure=True,
                  maximum_time_limit_gap=None,
                  wind_uncertainty_margin_mw=0.,progress_every=0,
                  tracking_policy="none",tracking_weight_usd_per_mwh=300.,
                  tracking_budget_mwh=.25,curtailment_enabled=False,
                  before_step=None, diagnostic_callback=None, diagnostic_stride=3,
                  warm_start_enabled=False,snapshot_at_utc=None,snapshot_path=None):
    """Run one economic study episode using the agreed supervisory correction.

    Notes:
      - episode_terminal_target_mwhth=None: finish at whatever SoC (normal HL
        terminal tracking remains active at each shorter horizon).
      - for fixed requested final SoC, the soft quadratic *replaces* the usual
        HL reference tracking when the physical episode end enters the horizon.
      - final penalty is an internal controller cost, NOT an ERCOT cash flow.
      - expects real 5min wind, 15min SPP, hourly DA data on UTC interval starts.
      - accepts only an already-cleared DA schedule supplied by the caller;
        multi-day DA publication cutoffs require a separately validated scheduler.
    """
    start=pd.Timestamp(start_utc)
    if start.tzinfo is None:raise ValueError('UTC issue requires timezone')
    start=start.tz_convert('UTC')
    cleared=pd.Timestamp(cleared_at_utc)
    if cleared.tzinfo is None or cleared.tz_convert('UTC')>start:
        raise ValueError('future DA clearing')
    if start.minute%15 or start.second or start.microsecond:
        raise ValueError('start must align to a quarter hour')
    if diagnostic_callback is not None and diagnostic_stride<1:raise ValueError('invalid diagnostic_stride')
    if n_steps<=0 or n_steps%12:
        raise ValueError('economic episodes must contain an integer number of complete hours')
    if episode_terminal_target_mwhth is not None and not 0<=episode_terminal_target_mwhth<=prm.capacity_mwhth:
        raise ValueError('episode terminal SoC outside tank bounds')
    def as_ser(x,name):
        s=pd.Series(x,dtype=float).sort_index()
        if not isinstance(s.index,pd.DatetimeIndex) or s.index.tz is None:
            raise ValueError(f'{name} requires timezone-aware timestamps')
        return s.tz_convert('UTC')
    past=as_ser(historic_wind_5min,'past wind')
    real=as_ser(realized_wind_5min,'realized wind')
    rt=as_ser(realized_rt15_price,'realized RT')
    q=as_ser(da_commitments_hourly,'DA schedules')
    da=as_ser(da_prices_hourly,'cleared DA prices')
    if len(past) and past.index.max()>=start:raise ValueError('wind history contains future')
    if len(real) and real.index.min()<start:raise ValueError('scored wind overlaps history')
    current=initial_state
    if curtailment_enabled and current.last_dispatched_wind_mw is None:
        prev_time=start-pd.Timedelta(minutes=5)
        if prev_time not in past.index:
            raise ValueError('wind ramp needs preceding 5-minute measured wind sample')
        current=replace(current,last_dispatched_wind_mw=float(past.loc[prev_time]))
    initial_energy=initial_state.energy_mwhth
    events=[];failed=0;last_rt_forecast=None;previous_executed_ptes_mw=0.
    accepted_time_limited=0
    previous_ll_solution=None
    if (snapshot_at_utc is None) != (snapshot_path is None):
        raise ValueError('snapshot time and destination must both be specified')
    snapshot_time=(pd.Timestamp(snapshot_at_utc).tz_convert('UTC')
                   if snapshot_at_utc is not None else None)
    ledger=SettlementLedger(q.to_dict(),da.to_dict(),rt.to_dict())
    for step in range(n_steps):
        t=start+pd.Timedelta(minutes=5*step)
        # A day-ahead bid may be issued at this time using the PREVIOUS
        # executed five-minute plant state. This callback is strictly causal.
        if before_step is not None:
            additions=before_step(t,current)
            if additions is not None:
                for h,newq in additions.items():
                    hour=pd.Timestamp(h).tz_convert('UTC')
                    if hour< t.floor('h') or hour in ledger.q_da_mw:
                        raise ValueError('DA commitment issued too late or overwritten')
                    ledger.q_da_mw[hour]=float(newq)
        remaining=n_steps-step
        # A FREE episode end is merely an accounting boundary: preserve a true
        # four-hour rolling LL horizon, even when it extends past the report end.
        # A specified episode-end target activates a *shortened* horizon only
        # when the target lies within four hours (soft, never a hard equality).
        horizon=min(48,remaining) if episode_terminal_target_mwhth is not None else 48
        measured_wind=pd.concat([past,real.loc[real.index<t]]).sort_index()
        if len(measured_wind) and measured_wind.index.max()>=t:
            raise AssertionError('wind lookahead leak')
        if t.minute%15==0:
            available_rt=rt.loc[rt.index+pd.Timedelta(minutes=15)<=t]
            if len(available_rt) and (available_rt.index+pd.Timedelta(minutes=15)).max()>t:
                raise AssertionError('RT lookahead leak')
            last_rt_forecast=np.asarray(rt_predictor(t,available_rt.copy(),da.copy(),
                                          measured_wind.copy()),float)
            if last_rt_forecast.shape!=(16,):raise ValueError('RT predictor must return 16 values')
            refresh_time=t
        if last_rt_forecast is None or t>=refresh_time+pd.Timedelta(minutes=15):
            raise ValueError('RT forecast stale')
        future_start=pd.date_range(t,periods=horizon,freq='5min',tz='UTC')
        all_wind=np.asarray(wind_predictor(t,measured_wind.copy()),float)
        if all_wind.shape!=(48,):raise ValueError('wind predictor must return 48 points')
        wind_hat=all_wind[:horizon]
        price_15=pd.Series(last_rt_forecast,index=pd.date_range(refresh_time,periods=16,
                                freq='15min',tz='UTC'))
        rt_hat=[]
        for f in future_start:
            bucket=f.floor('15min')
            if bucket in price_15:rt_hat.append(float(price_15[bucket]))
            else:rt_hat.append(float(last_rt_forecast[-1])) # 17th bucket; NEVER read future DA actual
        episode_end_in_horizon=(episode_terminal_target_mwhth is not None and remaining<=48)
        if episode_end_in_horizon:
            reference=float(episode_terminal_target_mwhth)
        else:
            reference=float(soc_target_at(t+pd.Timedelta(minutes=5*horizon)))
        # The accepted commercial position is carried as a constant within
        # every forecast 15-minute quarter. Future *unaccepted* DA hours
        # remain NaN: they are not invented or prematurely traded.
        accepted_q=np.asarray([ledger.q_da_mw.get(f.floor('h'),np.nan)
                               for f in future_start],float)
        quarter=t.floor('15min')
        count_done=int(ledger.count_5min.get(quarter,0))
        offset=int((t.minute%15)//5)
        if count_done!=offset:
            raise AssertionError('current settlement quarter executed-step mismatch')
        already_delivered=float(ledger.energy_mwh.get(quarter,0.))
        if snapshot_time is not None and t==snapshot_time:
            # Model inputs and *previously computed* binary seed only. The
            # actual wind for this future interval is not accessed here.
            import json
            from dataclasses import asdict
            from pathlib import Path
            candidate=previous_ll_solution
            payload={
                'timestamp_utc':t.isoformat(),
                'wind_forecast_mw':wind_hat.tolist(),
                'rt_forecast_usd_mwh':list(map(float,rt_hat)),
                'state':asdict(current), 'params':asdict(prm),
                'target_terminal_mwhth':reference,
                'accepted_da_mw':[None if np.isnan(v) else float(v) for v in accepted_q],
                'current_quarter_delivered_mwh':already_delivered,
                'quarter_offset_5min':offset,
                'tracking_policy':tracking_policy,
                'tracking_weight_usd_per_mwh':tracking_weight_usd_per_mwh,
                'tracking_budget_mwh':tracking_budget_mwh,
                'curtailment_enabled':curtailment_enabled,
                'terminal_beta_usd_per_mwhth2':terminal_beta_usd_per_mwhth2,
                'previous_modes':getattr(candidate,'modes',None),
                'previous_warm_start_seed':getattr(candidate,'warm_start_seed',None),
                'note':'Pre-decision causal snapshot; no realized wind of this interval',
            }
            destination=Path(snapshot_path)
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(payload,indent=2,allow_nan=False))
            print(f'LL identical-model cold/warm benchmark snapshot: {destination}',flush=True)
        sol=solve_ll(wind_hat,rt_hat,current,reference,prm,
                     terminal_beta_usd_per_mwhth2=terminal_beta_usd_per_mwhth2,
                     solver_backend=solver_backend,quadratic_tangents=quadratic_tangents,
                     wind_uncertainty_margin_mw=wind_uncertainty_margin_mw,
                     da_commitments_5min_mw=accepted_q,
                     current_quarter_delivered_mwh=already_delivered,
                     quarter_offset_5min=offset,tracking_policy=tracking_policy,
                     tracking_weight_usd_per_mwh=tracking_weight_usd_per_mwh,
                     tracking_budget_mwh=tracking_budget_mwh,
                     curtailment_enabled=curtailment_enabled,
                     time_limit_s=time_limit_s,mip_rel_gap=mip_rel_gap,
                     maximum_time_limit_gap=maximum_time_limit_gap,
                     **({'previous_solution':previous_ll_solution,
                         'warm_start_enabled':True} if warm_start_enabled else {}))
        if sol.charge_mw is None or not sol.success:
            failed+=1
            if stop_on_solver_failure:
                raise RuntimeError(f'LL solver did not certify requested tolerance at {t}: {sol.message}; gap={sol.mip_gap}')
            if current.mode in ('charge','discharge'):
                raise RuntimeError('no validated solver fallback from active mode')
            mode='off' if current.mode=='off' else 'standby';charge=discharge=0.
            cap_cmd=(min(prm.rated_wind_mw,current.last_dispatched_wind_mw)
                     if curtailment_enabled else None)
        else:
            mode=sol.modes[0]
            charge=float(sol.charge_mw[0]);discharge=float(sol.discharge_mw[0])
            cap_cmd=float(sol.wind_cap_mw[0]) if getattr(sol,'wind_cap_mw',None) is not None else None
        # Only a successfully optimized plan can initialize the next step.
        # A physical override is detected by the next step's state/mode check.
        previous_ll_solution=sol if sol.success and sol.modes is not None else None
        if getattr(sol,'accepted_time_limit_incumbent',False):
            accepted_time_limited+=1
            if accepted_time_limited<=5 or accepted_time_limited%25==0:
                print(f'WARNING LL {t.isoformat()}: accepted feasible Gurobi '
                      f'time-limit incumbent, gap={sol.mip_gap:.4%} '
                      f'(target {mip_rel_gap:.2%}, cap {maximum_time_limit_gap:.2%}); '
                      f'count={accepted_time_limited}',flush=True)
        # OBSERVABILITY ONLY: predictions and planned MPC trajectories are
        # recorded before the future realised observation is revealed.
        # This callback cannot modify either the solver command or plant state.
        if diagnostic_callback is not None and step % diagnostic_stride == 0:
            diagnostic_callback(t, current, wind_hat.copy(),
                                np.asarray(rt_hat, dtype=float), sol, reference)
        if t not in real.index:raise ValueError(f'missing realised wind {t}')
        # Forecasts issued at t are captured BEFORE the true t observation.
        forecast_poi=float(min(float(wind_hat[0]),cap_cmd if cap_cmd is not None else float(wind_hat[0]))-charge+discharge
            -(prm.standby_aux_mw if mode=='standby' else 0.))
        rt_prediction=float(rt_hat[0])
        # Realized interval data are only read AFTER the LL action has been chosen.
        wind_actual=float(real.loc[t])
        current,log=execute_step(current,mode,charge,discharge,wind_actual,prm,
                                 allow_safety_override=True,wind_dispatch_cap_mw=cap_cmd)
        ledger.record(t,log['poi_mw'])
        executed_ptes_mw=log['discharge_mw']-log['charge_mw']
        commanded_ptes_mw=log['commanded_discharge_mw']-log['commanded_charge_mw']
        commanded_step_change=commanded_ptes_mw-previous_executed_ptes_mw
        delta_ptes=executed_ptes_mw-previous_executed_ptes_mw
        previous_executed_ptes_mw=executed_ptes_mw
        events.append({'time_utc':t,'forecast_issued_utc':t,
                       'commanded_mode':mode,'actual_mode':current.mode,'mode':current.mode,
                       'forecast_wind_t0_mw':float(wind_hat[0]),'actual_wind_t0_mw':wind_actual,
                       'wind_dispatch_cap_mw':log['wind_dispatch_cap_mw'],
                       'curtailed_wind_mw':log['curtailed_wind_mw'],
                       'dispatched_wind_mw':log['dispatched_wind_mw'],
                       'wind_dispatch_change_mw':log['wind_dispatch_change_mw'],
                       'wind_downward_ramp_exception_mw':log['wind_downward_ramp_exception_mw'],
                       'forecast_poi_t0_mw':forecast_poi,
                       'forecast_rt15_usd_mwh':rt_prediction,
                       'actual_rt15_usd_mwh':float(rt.loc[t.floor('15min')]),
                       'commanded_charge_mw':log['commanded_charge_mw'],
                       'commanded_discharge_mw':log['commanded_discharge_mw'],
                       'charge_mw':log['charge_mw'],'discharge_mw':log['discharge_mw'],
                       'numerical_projection_mw':log['numerical_projection_mw'],
                       'charging_from_local_wind_mw':min(log['dispatched_wind_mw'],log['charge_mw']),
                       'charging_from_grid_mw':max(0.,log['charge_mw']-log['dispatched_wind_mw']),
                       'requested_poi_mw':log['requested_poi_mw'],
                       'physical_poi_mw':log['poi_mw'],
                       'discharge_clip_mw':log['discharge_clip_mw'],
                       'commanded_ptes_net_mw':commanded_ptes_mw,
                       'executed_ptes_net_mw':executed_ptes_mw,
                       'commanded_ptes_step_change_mw':commanded_step_change,
                       'regulatory_power_adjustment_mw':executed_ptes_mw-commanded_ptes_mw,
                       'regulatory_action':log['regulatory_action'],
                       'ptes_power_step_change_mw':delta_ptes,
                       'ptes_power_ramp_mw_per_min':delta_ptes/5.,
                       'tank_mwhth':current.energy_mwhth,
                       'off_elapsed_steps':current.off_elapsed_steps,
                       'll_horizon_steps':horizon,
                       'll_target_mwhth':reference,
                       'll_tracking_policy':tracking_policy,
                       'll_predicted_abs_tracking_mwh':getattr(sol,'predicted_tracking_abs_mwh',0.),
                       'll_predicted_weighted_tracking_mwh':getattr(sol,'predicted_weighted_tracking_mwh',0.),
                       'll_primary_stage_deviation_mwh':getattr(sol,'first_stage_tracking_mwh',None),
                       'da_commitment_mw':ledger.q_da_mw.get(t.floor('h'),np.nan),
                       'quarter_executed_energy_prior_mwh':already_delivered,
                       'episode_target_active':episode_end_in_horizon,
                       'rt_forecast_refreshed_utc':refresh_time,
                       'solver_status':sol.status,'solver_message':sol.message,
                       'solver_backend':sol.solver_backend,'mip_gap':sol.mip_gap,
                       'll_requested_gap_certified':getattr(sol,'certified_to_requested_gap',True),
                       'll_accepted_time_limit_incumbent':getattr(sol,'accepted_time_limit_incumbent',False),
                       'll_warm_start_submitted':getattr(sol,'warm_start_submitted',False),
                       'll_warm_start_variables':getattr(sol,'warm_start_variables',0),
                       'll_warm_start_note':getattr(sol,'warm_start_skip_reason','disabled'),
                       'll_solver_runtime_s':getattr(getattr(sol,'full_result',None),'runtime_seconds',np.nan),
                       'll_solver_nodes':getattr(getattr(sol,'full_result',None),'node_count',np.nan),
                       'quadratic_error_bound_usd':sol.max_quadratic_approx_error_usd,
                       'safety_override':log['safety_override']})
        if progress_every and ((step+1)%progress_every==0):
            print(f'LL progress {step+1}/{n_steps}, time={t.isoformat()}, mode={current.mode}, SoC={current.energy_mwhth:.1f} MWhth',flush=True)
    cash=ledger.full_cashflow()
    ev=pd.DataFrame(events)
    terminal_error=(None if episode_terminal_target_mwhth is None
                    else current.energy_mwhth-episode_terminal_target_mwhth)
    soft_penalty=(0. if terminal_error is None else terminal_beta_usd_per_mwhth2*terminal_error**2)
    # In addition to cashflow, report real signed 15-minute DA errors (MWh).
    e_quarter=(ev.assign(quarter=ev.time_utc.dt.floor('15min'))
               .groupby('quarter')['physical_poi_mw'].sum()/12.)
    true_abs_da=sum(abs(float(e)-float(ledger.q_da_mw[k.floor('h')])/4.)
                    for k,e in e_quarter.items())
    return RunResult(ev,cash,current,failed,initial_energy,episode_terminal_target_mwhth,
                     terminal_error,soft_penalty,
                     int(ev['safety_override'].sum()),
                     int((ev['regulatory_action']=='subminimum_forced_off').sum()),
                     float(ev['discharge_clip_mw'].sum()/12),
                     float(ev['ptes_power_step_change_mw'].abs().max()),
                     float(ev['curtailed_wind_mw'].sum()/12.),
                     float(ev['wind_dispatch_change_mw'].abs().max()),
                     float(true_abs_da),
                     int((ev['wind_dispatch_change_mw']>prm.wind_dispatch_ramp_mw_per_5min+1e-5).sum()),
                     int((ev['wind_downward_ramp_exception_mw']>1e-5).sum()),
                     float(ev['wind_dispatch_change_mw'].max()),
                     accepted_time_limited,
                     float(pd.to_numeric(ev['mip_gap'],errors='coerce').max()))
