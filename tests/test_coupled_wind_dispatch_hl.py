"""Regression coverage for the added economically dispatchable HL wind variable.

These tests use tiny synthetic DEVELOPMENT scenarios only. No blind holdout data.
"""
import numpy as np
import pytest
from ptes_mpc.hl_market import solve_hl_market


def _small(curt, risk_scale=1.):
    w=np.full((2,24),105.,float)
    price=np.full((2,24),-25.,float)
    return solve_hl_market(w,price,[.5,.5],0.,np.ones(24),np.full(24,.5),
        solver_backend='highs',time_limit_s=25.,mip_rel_gap=.05,
        hl_curtailment_enabled=curt,hl_risk_scale=risk_scale)


def test_hl_optional_wind_curtailment_keeps_storage_physics_and_changes_dispatch():
    old=_small(False)
    new=_small(True)
    assert old.success and new.success
    assert old.wind_curtailment_enabled is False
    assert new.wind_curtailment_enabled is True
    assert len(new.reference_mwhth)==24
    assert len(new.commitment_mw)==24
    assert np.allclose(old.expected_wind_dispatch_mw,105.)
    assert np.all(new.expected_wind_dispatch_mw>=-1e-7)
    assert np.all(new.expected_wind_dispatch_mw<=105.+1e-6)
    # Curtailment becomes financially rational when all prospective prices are negative.
    assert new.expected_wind_dispatch_mw.mean()<old.expected_wind_dispatch_mw.mean()-5
    assert new.objective_proxy_usd>=old.objective_proxy_usd-1e-4


def test_hl_risk_scale_validation():
    with pytest.raises(ValueError,match='risk scale'):
        _small(True,-1.)


def test_rt_forecast_tail_never_peeks_at_uncleared_future_DA(monkeypatch):
    """At minute 05 the 4h LL horizon includes the 17th price quarter."""
    import pandas as pd
    from types import SimpleNamespace
    from ptes_mpc import harness
    from ptes_mpc.physics import PlantState
    from run_multiday import local_midnight
    start=local_midnight('2025-12-18')
    past_ix=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min',tz='UTC')
    real_ix=pd.date_range(start,periods=12,freq='5min',tz='UTC')
    rt_ix=pd.date_range(start,periods=32,freq='15min',tz='UTC')
    da_ix=pd.date_range(start,periods=16,freq='h',tz='UTC')
    calls=[]
    def fake_ll(wind_hat, rt_hat, state, target, prm, **kw):
        calls.append(np.asarray(rt_hat))
        return SimpleNamespace(charge_mw=np.zeros(48),discharge_mw=np.zeros(48),
                modes=['off']*48,success=True,status=2,message='mock',mip_gap=0.,
                solver_backend='mock',max_quadratic_approx_error_usd=0.)
    monkeypatch.setattr(harness,'solve_ll',fake_ll)
    harness.simulate_5min(start_utc=start,n_steps=12,
        historic_wind_5min=pd.Series(50.,index=past_ix),
        realized_wind_5min=pd.Series(50.,index=real_ix),
        realized_rt15_price=pd.Series(5.,index=rt_ix),
        da_commitments_hourly=pd.Series(50.,index=da_ix),
        # A sentinel for a future hourly DA price that the LL must NOT read.
        da_prices_hourly=pd.Series(9999.,index=da_ix),
        cleared_at_utc=start-pd.Timedelta(hours=2),
        wind_predictor=lambda t,h:np.full(48,50.),
        rt_predictor=lambda t,a,d,w:np.full(16,7.),
        initial_state=PlantState(0.,'off',6),
        soc_target_at=lambda t:0.,tracking_policy='none',
        solver_backend='highs')
    assert len(calls)==12
    assert np.allclose(calls[1],7.)
    assert not any((call==9999.).any() for call in calls)
