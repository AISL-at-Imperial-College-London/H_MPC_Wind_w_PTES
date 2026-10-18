"""New development-week guardrails and passive LL trace validation."""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from run_multiday import validate_days, local_midnight
from ptes_mpc import harness
from ptes_mpc.physics import PlantState


def test_new_seven_day_week_is_development_only():
    assert validate_days('2025-12-08',7)==[
        '2025-12-08','2025-12-09','2025-12-10','2025-12-11',
        '2025-12-12','2025-12-13','2025-12-14']
    with pytest.raises(ValueError):
        validate_days('2026-02-09',7)  # protected evaluation period
    with pytest.raises(ValueError):
        validate_days('2025-12-07',7)  # no prior 48h wind-only baseline


def test_diagnostic_callback_is_pre_realisation_and_passive(monkeypatch):
    start=local_midnight('2025-12-08')
    prior=pd.Series(100.,index=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min'))
    realised=pd.Series(105.,index=pd.date_range(start,periods=12,freq='5min'))
    q=pd.Series(100.,index=pd.date_range(start,periods=24,freq='h'))
    da=pd.Series(10.,index=q.index)
    rt=pd.Series(8.,index=pd.date_range(start,periods=4,freq='15min'))
    trace=[]
    def solver(wind, price, state, target, prm, **kwargs):
        n=len(wind)
        return SimpleNamespace(charge_mw=np.zeros(n),discharge_mw=np.zeros(n),
                               tank_mwhth=np.zeros(n),modes=['off']*n,
                               success=True,status=2,message='mocked',
                               solver_backend='test',mip_gap=0.,
                               max_quadratic_approx_error_usd=0.)
    monkeypatch.setattr(harness,'solve_ll',solver)
    def wind_predictor(t,h):
        assert (h.index<t).all()
        return np.full(48,100.)
    def trace_cb(t,state,w,p,sol,reference):
        assert state.energy_mwhth==0.
        assert np.all(w==100.)
        assert np.all(p==8.)
        trace.append(t)
    result=harness.simulate_5min(start_utc=start,n_steps=12,
        historic_wind_5min=prior,realized_wind_5min=realised,
        realized_rt15_price=rt,da_commitments_hourly=q,da_prices_hourly=da,
        cleared_at_utc=start-pd.Timedelta(hours=5),
        wind_predictor=wind_predictor,
        rt_predictor=lambda *args:np.full(16,8.),
        initial_state=PlantState(0.,'off',6),soc_target_at=lambda _:0.,
        diagnostic_callback=trace_cb,diagnostic_stride=3)
    assert trace==[start+pd.Timedelta(minutes=i) for i in (0,15,30,45)]
    assert len(result.events)==12
    assert (result.events.actual_wind_t0_mw==105.).all()
    assert (result.events.forecast_wind_t0_mw==100.).all()
    assert result.solve_failures==0
