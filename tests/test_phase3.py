"""Phase 3: as-of economics, no hidden curtailment, fixed economics and free-horizon."""
import numpy as np
import pandas as pd
import pytest
from ptes_mpc.physics import PlantParams,PlantState,execute_step
from ptes_mpc.hl_market import solve_hl_market,estimate_spread_from_past
from ptes_mpc.settlement import SettlementLedger
from run_development_day import available_da,clearing_asof,soc_lookup

def test_hl_equalizes_first_day_terminal_and_physical_poi():
    p=PlantParams()
    w=np.array([[50]*24,[160]*24],float)
    price=np.array([[30]*24,[20]*24],float)
    sol=solve_hl_market(w,price,[.5,.5],0.,np.zeros(24),np.full(24,7.),time_limit_s=10,mip_rel_gap=.08)
    assert sol.commitment_mw is not None
    assert np.all(abs(sol.commitment_mw)<=230+1e-7)
    assert len(sol.reference_mwhth)==24
    assert np.ptp(sol.scenario_end_day1_mwhth)<1e-5


def test_asof_da_no_nextday_at_midnight():
    t=pd.Timestamp('2025-12-18T06:00:00Z') # Chicago local midnight
    days=pd.date_range(t-pd.Timedelta(hours=24),periods=50,freq='h',tz='UTC')
    prices=pd.Series(np.arange(len(days),dtype=float),index=days)
    known=available_da(t,prices)
    assert t.floor('h') in known.index
    assert (t+pd.Timedelta(hours=24)) not in known.index
    assert clearing_asof(t)<=t


def test_linear_reference_interpolation():
    start=pd.Timestamp('2025-12-18T06:00:00Z')
    fn=soc_lookup(start,200.,np.array([400.,600.]))
    assert fn(start)==pytest.approx(200.)
    assert fn(start+pd.Timedelta(minutes=30))==pytest.approx(300.)
    assert fn(start+pd.Timedelta(hours=2))==pytest.approx(600.)


def test_minload_rounding_only():
    p=PlantParams()
    s=PlantState(500,'charge',0)
    ns,log=execute_step(s,'charge',p.charge_min_mw-3e-5,0,100,p)
    assert ns.mode=='charge'
    assert log['numerical_projection_mw']==pytest.approx(3e-5,abs=1e-9)
    with pytest.raises(ValueError,match='charging part-load'):
        execute_step(s,'charge',p.charge_min_mw-.01,0,100,p)


def test_two_settlement_with_negative_price_signed_import():
    from datetime import datetime,timedelta,timezone
    t=datetime(2025,12,18,6,tzinfo=timezone.utc)
    q={t:100.};da={t:-5.};rt={t+timedelta(minutes=i*15):-10. for i in range(4)}
    ledger=SettlementLedger(q,da,rt)
    for j in range(12):ledger.record(t+timedelta(minutes=j*5),-20.)
    cash=ledger.full_cashflow()
    assert cash['da_usd']==pytest.approx(-500.)
    assert cash['rt_usd']==pytest.approx(1200.)
    assert cash['total_usd']==pytest.approx(700.)


def test_historical_spread_rejects_future():
    start=pd.Timestamp('2025-12-18T06:00:00Z')
    idx=pd.date_range(end=start+pd.Timedelta(hours=1),periods=28*96+8,freq='15min',tz='UTC')
    h=idx.floor('h').unique()
    da=pd.Series(10.,index=h)
    a=pd.Series(11.,index=idx)
    target=pd.date_range(start,periods=84,freq='h',tz='UTC')
    # Deliberately use data *after* the issue; selection must drop them.
    mean,risk=estimate_spread_from_past(rt15=a,da_hourly=da,decision_utc=start,delivery_times_utc=target)
    assert np.allclose(mean,1.)
    b=a.copy();b.loc[b.index>=start]=1e6
    mean2,risk2=estimate_spread_from_past(rt15=b,da_hourly=da,decision_utc=start,delivery_times_utc=target)
    assert np.allclose(mean,mean2)
    assert np.allclose(risk,risk2)
