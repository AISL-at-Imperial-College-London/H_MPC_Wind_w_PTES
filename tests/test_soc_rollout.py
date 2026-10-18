"""Unit tests for the bid-to-delivery SoC rollout (no solver, no xgboost)."""
import numpy as np
import pandas as pd
import pytest
from ptes_mpc.physics import PlantParams, PlantState
from soc_rollout import gap_wind_forecast, rollout_delivery_soc, forecast_rollout

PRM = PlantParams()
T0 = pd.Timestamp('2025-12-08 15:30', tz='UTC')        # 09:30 CST
T1 = pd.Timestamp('2025-12-09 06:00', tz='UTC')        # delivery midnight CST
N = 174


def _q(val):
    return {h: val for h in pd.date_range(T0.floor('h'), T1, freq='h', tz='UTC')}


def test_gap_wind_relaxes_to_anchor():
    p = gap_wind_forecast(T0, N, np.full(48, 200.), 50., relax_hours=2.)
    assert p.shape == (N,) and np.allclose(p[:48], 200.) and abs(p[-1] - 50.) < 1.


def test_surplus_wind_charges_and_shortfall_discharges():
    up = rollout_delivery_soc(issue_utc=T0, delivery_utc=T1, state=PlantState(500., 'standby', 0),
                              commitments_hourly=_q(20.), da_prices_hourly=_q(30.),
                              wind_path=np.full(N, 150.), wind_source='test')
    dn = rollout_delivery_soc(issue_utc=T0, delivery_utc=T1, state=PlantState(500., 'standby', 0),
                              commitments_hourly=_q(150.), da_prices_hourly=_q(30.),
                              wind_path=np.full(N, 20.), wind_source='test')
    assert up.predicted_soc_mwhth > 500. > dn.predicted_soc_mwhth
    for r in (up, dn):
        tr = r.trajectory
        assert (tr.charge_mw <= PRM.charge_max_mw + 1e-6).all()
        assert (tr.discharge_mw <= PRM.discharge_max_mw + 1e-6).all()
        assert (tr.poi_mw.abs() <= PRM.poi_limit_mw + 1e-6).all()
        assert tr.tank_mwhth.between(0, PRM.capacity_mwhth).all()


def test_exact_tracking_leaves_state_nearly_unchanged():
    r = rollout_delivery_soc(issue_utc=T0, delivery_utc=T1, state=PlantState(800., 'off', 6),
                             commitments_hourly=_q(100.), da_prices_hourly=_q(30.),
                             wind_path=np.full(N, 100.), wind_source='test')
    assert abs(r.predicted_soc_mwhth - 800. * PRM.daily_retention ** (14.5 / 24)) < 1e-6


def test_future_measurements_rejected():
    meas = pd.Series(100., index=pd.date_range(T0 - pd.Timedelta(hours=5), T0, freq='5min', tz='UTC'))
    with pytest.raises(ValueError, match='at/after the bid time'):
        forecast_rollout(issue_utc=T0, delivery_utc=T1, state=PlantState(0.),
                         measured_wind_5min=meas, short_term_wind_predict=lambda t, m: np.full(48, 100.),
                         anchor_wind_mw=100., commitments_hourly=_q(0.), da_prices_hourly=_q(30.))


def test_missing_commitment_rejected():
    q = _q(50.); q.pop(T0.floor('h') + pd.Timedelta(hours=3))
    with pytest.raises(ValueError, match='no accepted DA commitment'):
        rollout_delivery_soc(issue_utc=T0, delivery_utc=T1, state=PlantState(0.),
                             commitments_hourly=q, da_prices_hourly=_q(30.),
                             wind_path=np.full(N, 50.), wind_source='test')


def test_rollout_curtailment_respects_upward_ramp_and_never_creates_wind():
    w = np.full(N, 120.)
    w[:4] = [200., 200., 20., 200.]
    start = PlantState(800., 'off', 6, last_dispatched_wind_mw=100.)
    r = rollout_delivery_soc(issue_utc=T0, delivery_utc=T1, state=start,
                             commitments_hourly=_q(110.), da_prices_hourly=_q(30.),
                             wind_path=w, wind_source='synthetic',
                             curtailment_enabled=True)
    delivered = r.trajectory.dispatched_wind_mw.to_numpy(float)
    assert (delivered <= w + 1e-8).all()
    assert (np.diff(np.r_[100., delivered]) <= 40.00001).all()
    assert delivered[2] == 20.  # Wind resource collapses; cannot fabricate 80 MW.
    assert r.wind_downward_ramp_exceptions >= 1
    assert r.curtailed_wind_mwh > 0
    assert np.isclose(r.curtailed_wind_mwh, r.trajectory.curtailed_wind_mw.sum()/12.)


def test_rollout_requires_known_previous_wind_when_ramp_enabled():
    import pytest
    with pytest.raises(ValueError, match='preceding realised dispatched wind'):
        rollout_delivery_soc(issue_utc=T0, delivery_utc=T1,
            state=PlantState(100., 'off', 6), commitments_hourly=_q(80.),
            da_prices_hourly=_q(10.), wind_path=np.full(N, 100.),
            wind_source='synthetic', curtailment_enabled=True)


def test_causal_rollout_oracle_same_path_same_initial_state():
    from soc_rollout import oracle_rollout
    start=PlantState(500.,'standby',0,last_dispatched_wind_mw=120.)
    windpath=np.full(N, 100.)
    series=pd.Series(windpath,index=pd.date_range(T0,periods=N,freq='5min',tz='UTC'))
    common=dict(issue_utc=T0,delivery_utc=T1,state=start,
                commitments_hourly=_q(80.),da_prices_hourly=_q(30.),
                curtailment_enabled=True)
    a=rollout_delivery_soc(**common,wind_path=windpath,wind_source='synthetic')
    b=oracle_rollout(**common,realised_wind_5min=series)
    assert np.isclose(a.predicted_soc_mwhth,b.predicted_soc_mwhth)
    assert np.isclose(a.curtailed_wind_mwh,b.curtailed_wind_mwh)


def test_gap_wind_rejects_invalid_relaxation_parameter():
    with pytest.raises(ValueError, match='relaxation time'):
        gap_wind_forecast(T0,N,np.full(48,100.),60.,relax_hours=0.)
