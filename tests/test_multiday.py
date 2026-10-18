import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from ptes_mpc import harness
from ptes_mpc.physics import PlantState,PlantParams
from run_multiday import validate_days, local_midnight


def test_study_window_protects_holdouts():
    assert validate_days('2025-12-18',3)==['2025-12-18','2025-12-19','2025-12-20']
    for day,duration in [('2026-02-09',3),('2025-12-18',1),('2025-12-18',8)]:
        with pytest.raises(ValueError):validate_days(day,duration)


def test_harness_accepts_next_day_bids_only_before_delivery(monkeypatch):
    start=local_midnight('2025-12-18')
    step_count=25*12  # cross a day boundary and settle 25 full hours
    past_index=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min',tz='UTC')
    real_index=pd.date_range(start,periods=step_count,freq='5min',tz='UTC')
    k_index=pd.date_range(start-pd.Timedelta(hours=1),periods=step_count//3+8,freq='15min',tz='UTC')
    h_index=pd.date_range(start-pd.Timedelta(hours=1),periods=32,freq='h',tz='UTC')
    q=pd.Series(20.,index=h_index[1:25])  # first 24h only
    issues=[]
    def callback(t,state):
        if t==start+pd.Timedelta(hours=9,minutes=30):
            issues.append((t,state.energy_mwhth))
            return pd.Series(50.,index=h_index[25:])
        return None
    def fake_solver(wind_hat,rt_hat,state,reference,prm,**kw):
        return SimpleNamespace(charge_mw=np.zeros(48),discharge_mw=np.zeros(48),modes=['off']*48,
            success=True,status=2,message='mocked',solver_backend='mock',mip_gap=0.,max_quadratic_approx_error_usd=0.)
    monkeypatch.setattr(harness,'solve_ll',fake_solver)
    def predictor(t,hist):
        assert len(hist) and hist.index.max()<t
        return np.full(48,50.)
    r=harness.simulate_5min(start_utc=start,n_steps=step_count,
        historic_wind_5min=pd.Series(50.,index=past_index),
        realized_wind_5min=pd.Series(50.,index=real_index),
        realized_rt15_price=pd.Series(12.,index=k_index),
        da_commitments_hourly=q,da_prices_hourly=pd.Series(15.,index=h_index),
        cleared_at_utc=start-pd.Timedelta(hours=6),
        wind_predictor=predictor,rt_predictor=lambda t,rt,da,w:np.full(16,12.),
        initial_state=PlantState(0.,'off',6),soc_target_at=lambda t:0.,
        solver_backend='highs',before_step=callback)
    assert issues==[(start+pd.Timedelta(hours=9,minutes=30),0.)]
    assert len(r.events)==step_count
    assert r.cashflow['scheduled_mwh']==24*20+50
    assert abs(r.cashflow['phys_mwh']-25*50)<1e-5


def test_rejects_retroactive_da_schedule(monkeypatch):
    start=local_midnight('2025-12-18')
    hist=pd.Series(100.,index=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min',tz='UTC'))
    real=pd.Series(100.,index=pd.date_range(start,periods=12,freq='5min',tz='UTC'))
    hidx=pd.date_range(start,periods=24,freq='h',tz='UTC')
    q=pd.Series(10.,index=hidx)
    with pytest.raises(ValueError,match='too late or overwritten'):
        harness.simulate_5min(start_utc=start,n_steps=12,historic_wind_5min=hist,
          realized_wind_5min=real,
          realized_rt15_price=pd.Series(10.,index=pd.date_range(start,periods=4,freq='15min',tz='UTC')),
          da_commitments_hourly=q,da_prices_hourly=pd.Series(10.,index=hidx),
          cleared_at_utc=start-pd.Timedelta(hours=6),
          wind_predictor=lambda *args:np.ones(48)*100.,rt_predictor=lambda *args:np.ones(16)*10.,
          initial_state=PlantState(0.,'off',6),soc_target_at=lambda t:0.,
          before_step=lambda t,state:{start:0.})
