"""Unit/integration checks for single-shot tracking MIQP and 40 MW wind-power ramp.
All test dates stay within the development partition; no blind holdouts are read.
"""
import numpy as np
import pandas as pd
import pytest
from ptes_mpc.physics import PlantParams,PlantState,execute_step
from ptes_mpc.ll_mpc import solve_ll


def test_wind_cap_upward_limit_and_unavoidable_drop():
    p=PlantParams()
    assert p.wind_dispatch_ramp_mw_per_5min==40.
    s=PlantState(500.,'off',6,100.)
    s,l=execute_step(s,'off',0,0,200,p,wind_dispatch_cap_mw=140.)
    assert l['dispatched_wind_mw']==140.
    assert l['wind_dispatch_change_mw']==40.
    assert l['curtailed_wind_mw']==60.
    with pytest.raises(ValueError,match='output cap ramp'):
        execute_step(s,'off',0,0,200,p,wind_dispatch_cap_mw=181.)
    s,l=execute_step(s,'off',0,0,9,p,wind_dispatch_cap_mw=100.)
    assert l['dispatched_wind_mw']==9. and l['wind_downward_ramp_exception_mw']==91.
    # A later wind recovery cannot jump straight to 200 MW.
    s,l=execute_step(s,'off',0,0,200,p,wind_dispatch_cap_mw=49.)
    assert l['dispatched_wind_mw']==49.
    assert l['wind_dispatch_change_mw']==40.


def test_single_shot_full_quarter_tracks_at_cap_rate():
    p=PlantParams()
    s=PlantState(0.,'off',0,180.)  # machine cannot start, only dispatchable wind
    sol=solve_ll([180.]*3,[35.]*3,s,0.,p,
        da_commitments_5min_mw=[160.]*3,tracking_policy='weighted',
        tracking_weight_usd_per_mwh=300.,curtailment_enabled=True,
        terminal_beta_usd_per_mwhth2=.01,solver_backend='highs',
        time_limit_s=8,quadratic_tangents=21)
    assert sol.success,sol.message
    assert sol.first_stage_tracking_mwh is None  # one optimization solve
    assert sol.wind_cap_mw is not None
    assert abs(sol.poi_forecast_mw.sum()/12.-40.)<1e-4
    assert sol.wind_cap_mw[0]<=220.+1e-6
    assert np.max(np.abs(np.diff(sol.wind_cap_mw)-np.diff(sol.dispatched_wind_forecast_mw)))<=40+1e-6


def test_partial_quarter_includes_executed_poi_mwh():
    p=PlantParams()
    s=PlantState(0.,'off',0,180.)
    sol=solve_ll([180.,180.],[35.,35.],s,0.,p,
        da_commitments_5min_mw=[160.,160.],quarter_offset_5min=1,
        current_quarter_delivered_mwh=15.,tracking_policy='weighted',
        curtailment_enabled=True,terminal_beta_usd_per_mwhth2=0.,
        solver_backend='highs',time_limit_s=8,quadratic_tangents=21)
    assert sol.success,sol.message
    assert abs(15.+sum(sol.poi_forecast_mw)/12.-40.)<1e-4


def test_forecast_wind_drop_has_no_fictitious_generation():
    p=PlantParams()
    s=PlantState(50.,'off',0,180.)
    sol=solve_ll([10.,190.,220.],[50.]*3,s,50.,p,
        da_commitments_5min_mw=[150.]*3,tracking_policy='weighted',
        curtailment_enabled=True,terminal_beta_usd_per_mwhth2=.01,
        solver_backend='highs',quadratic_tangents=21,time_limit_s=8.)
    assert sol.success,sol.message
    assert sol.dispatched_wind_forecast_mw[0]<=10.+1e-7
    assert sol.dispatched_wind_forecast_mw[1]<=sol.dispatched_wind_forecast_mw[0]+40.+1e-5
    assert np.all(sol.dispatched_wind_forecast_mw<=np.array([10,190,220])+1e-7)


def test_harness_energy_signed_settlement_and_causal_feedback(monkeypatch):
    from types import SimpleNamespace
    from ptes_mpc import harness
    start=pd.Timestamp('2025-12-18T06:00:00Z')
    prior=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min')
    active=pd.date_range(start,periods=12,freq='5min')
    rt_times=pd.date_range(start-pd.Timedelta(minutes=15),periods=20,freq='15min')
    da_times=pd.date_range(start,periods=8,freq='h')
    solver_calls=[]
    def mock_solver(w,lam,state,reference,prm,**kw):
        solver_calls.append((state.last_dispatched_wind_mw,kw['current_quarter_delivered_mwh']))
        return SimpleNamespace(charge_mw=np.zeros(48),discharge_mw=np.zeros(48),
            wind_cap_mw=np.full(48,175.),modes=['off']*48,
            success=True,status=2,message='mock',solver_backend='mock',
            mip_gap=0.,max_quadratic_approx_error_usd=0.)
    monkeypatch.setattr(harness,'solve_ll',mock_solver)
    r=harness.simulate_5min(start_utc=start,n_steps=12,
        historic_wind_5min=pd.Series(200.,index=prior),
        realized_wind_5min=pd.Series(200.,index=active),
        realized_rt15_price=pd.Series(20.,index=rt_times),
        da_commitments_hourly=pd.Series(175.,index=da_times[:1]),
        da_prices_hourly=pd.Series(30.,index=da_times),
        cleared_at_utc=start-pd.Timedelta(hours=1),
        wind_predictor=lambda t,obs:np.full(48,200.),
        rt_predictor=lambda t,prices,da,w:np.full(16,20.),
        initial_state=PlantState(0.,'off',6),soc_target_at=lambda t:0.,
        solver_backend='highs',tracking_policy='weighted',curtailment_enabled=True)
    assert r.curtailed_wind_mwh==pytest.approx(25.)
    assert r.cashflow['phys_mwh']==pytest.approx(175.)
    assert r.cashflow['da_usd']==pytest.approx(5250.)
    assert r.cashflow['rt_usd']==pytest.approx(0.)
    assert r.abs_da_deviation_mwh<1e-8
    assert solver_calls[0][0]==200. and solver_calls[1][0]==175.
    assert solver_calls[1][1]==pytest.approx(175./12.)
    assert r.positive_wind_ramp_violations==0
