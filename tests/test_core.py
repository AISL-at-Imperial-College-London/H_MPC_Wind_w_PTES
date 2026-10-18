from datetime import datetime,timedelta,timezone
import numpy as np
import pytest
from ptes_mpc import PlantParams,PlantState,SettlementLedger,execute_step
from ptes_mpc.physics import thermal_charge_mw,thermal_discharge_mw,pwa_breakpoints

def test_physics_bounds_and_derate():
    p=PlantParams()
    assert thermal_charge_mw(p.charge_min_mw,p)>0
    assert thermal_discharge_mw(p.discharge_min_mw,p)>0
    s=PlantState(1000,'off',6)
    s,log=execute_step(s,'standby',0,0,100,p)
    assert log['poi_mw']==100-p.standby_aux_mw
    with pytest.raises(ValueError): execute_step(s,'discharge',0,p.discharge_max_mw,100,p)
    s,log=execute_step(s,'discharge',0,p.discharge_min_mw,30,p)
    assert s.energy_mwhth<1000
    with pytest.raises(ValueError): execute_step(s,'charge',p.charge_min_mw,0,30,p)

def test_off_clock():
    p=PlantParams()
    s=PlantState(1200,'charge',0)
    s,_=execute_step(s,'off',0,0,10,p)
    with pytest.raises(ValueError):execute_step(s,'standby',0,0,10,p)
    for _ in range(5):s,_=execute_step(s,'off',0,0,10,p)
    assert s.off_elapsed_steps==6
    execute_step(s,'standby',0,0,10,p)

def test_pwa_accuracy_and_energy():
    p=PlantParams()
    for mode in ['charge','discharge']:
        xx,yy=pwa_breakpoints(mode,p)
        assert len(xx)==6 and np.min(np.diff(xx))>0
        fun=thermal_charge_mw if mode=='charge' else thermal_discharge_mw
        err=max(abs(np.interp(x,xx,yy)-fun(x,p)) for x in np.linspace(xx[0],xx[-1],100))
        assert err<6.0

def test_quarter_and_hour_settlement():
    base=datetime(2025,7,14,10,tzinfo=timezone.utc)
    q={base:50.};da={base:30.}
    rts={base+timedelta(minutes=15*i):40.+i for i in range(4)}
    ledger=SettlementLedger(q,da,rts)
    for i in range(12):ledger.record(base+timedelta(minutes=5*i),90.)
    cash=ledger.full_cashflow()
    assert abs(cash['da_usd']-1500)<1e-7
    assert abs(cash['rt_usd']-sum((90-50)*.25*(40+i) for i in range(4)))<1e-7
    assert abs(cash['phys_mwh']-90.)<1e-8
    with pytest.raises(ValueError): ledger.record(base,90)
    # Import and negative prices are signed without special-case shortcuts.
    q2={base:-10};da2={base:-5};rts2={k:-20 for k in rts}
    l2=SettlementLedger(q2,da2,rts2)
    for i in range(12):l2.record(base+timedelta(minutes=5*i),-15.)
    c=l2.full_cashflow()
    assert abs(c['total_usd']-(50+100))<1e-8

def test_agreed_zero_below_min_clipping():
    p=PlantParams()
    s=PlantState(1500,'discharge',0)
    with pytest.raises(ValueError,match='POI'):
        execute_step(s,'discharge',0,p.discharge_min_mw,230,p,
                     allow_safety_override=False)
    nxt,log=execute_step(s,'discharge',0,p.discharge_min_mw,230,p)
    assert log['requested_poi_mw']>p.poi_limit_mw
    assert log['regulatory_action']=='subminimum_forced_off'
    assert log['discharge_mw']==0 and log['discharge_clip_mw']==p.discharge_min_mw
    assert log['poi_mw']==p.poi_limit_mw
    assert nxt.mode=='off' and nxt.off_elapsed_steps==1
    assert nxt.energy_mwhth==pytest.approx(s.energy_mwhth*p.daily_retention**(1/12/24))
    with pytest.raises(ValueError,match='minimum off'):
        execute_step(nxt,'standby',0,0,230,p)

def test_partial_clipping_respects_floor_and_actual_energy():
    p=PlantParams(); s=PlantState(1200,'discharge',0)
    nxt,log=execute_step(s,'discharge',0,100,160,p)
    assert log['regulatory_action']=='partial_discharge_backdown'
    assert log['discharge_mw']==pytest.approx(70)
    assert log['discharge_clip_mw']==pytest.approx(30)
    assert log['poi_mw']==pytest.approx(230)
    assert nxt.mode=='discharge'
    assert nxt.energy_mwhth==pytest.approx(
        1200*p.daily_retention**(1/12/24)-thermal_discharge_mw(70,p)/12)
    nxt2,log2=execute_step(s,'discharge',0,100,210,p)
    assert nxt2.mode=='off' and log2['discharge_mw']==0
    assert log2['poi_mw']==210


def test_solve_ll_smoke():
    from ptes_mpc import solve_ll
    p=PlantParams()
    w=np.full(48,120.)
    price=np.full(48,0.)
    out=solve_ll(w,price,PlantState(1100.,'off',6),1100.,p,time_limit_s=12,mip_rel_gap=.1)
    assert out.charge_mw is not None, out.message
    assert len(out.charge_mw)==48 and len(out.modes)==48
    assert np.max(np.abs(out.poi_forecast_mw))<=230+1e-6
    assert out.tank_mwhth.min()>0

def test_solve_hl_smoke():
    from ptes_mpc import solve_hl
    p=PlantParams()
    w=np.array([[100.]*24,[120.]*24])
    price=np.array([[20.]*24,[25.]*24])
    out=solve_hl(w,price,[.5,.5],1000.,p,time_limit_s=12,mip_rel_gap=.1)
    assert out.commitment_mw is not None,out.message
    assert out.commitment_mw.shape==(24,)
    assert out.reference_mwhth.shape==(24,)
    assert abs(out.commitment_mw).max()<230.0001

def test_causal_harness_one_hour():
    import pandas as pd
    from ptes_mpc.harness import simulate_5min
    start=pd.Timestamp('2025-07-14 10:00',tz='UTC')
    pastidx=pd.date_range(end=start-pd.Timedelta(minutes=5),periods=49,freq='5min',tz='UTC')
    futidx=pd.date_range(start,periods=12,freq='5min',tz='UTC')
    rtidx=pd.date_range(end=start-pd.Timedelta(minutes=15),periods=16,freq='15min',tz='UTC')
    future_rtidx=pd.date_range(start,periods=4,freq='15min',tz='UTC')
    daidx=pd.date_range(start.floor('h'),periods=6,freq='h',tz='UTC')
    rt_series=pd.Series(0.,index=rtidx.union(future_rtidx))
    past=pd.Series(100.,index=pastidx)
    actual=pd.Series(100.,index=futidx)
    observed=[];qtimes=[]
    def wf(t,x):
        observed.append((t,x.index.max()))
        return np.full(48,float(x.iloc[-1]))
    def pf(t,x,cleared,wind_past):
        qtimes.append((t,x.index.max()))
        assert wind_past.index.max()<t
        return np.zeros(16)
    rr=simulate_5min(start_utc=start,n_steps=12,
        historic_wind_5min=past,realized_wind_5min=actual,
        realized_rt15_price=rt_series,da_commitments_hourly=pd.Series(80.,index=daidx),
        da_prices_hourly=pd.Series(30.,index=daidx),
        cleared_at_utc=start-pd.Timedelta(hours=4),
        wind_predictor=wf,rt_predictor=pf,
        initial_state=PlantState(1200.,'off',6),soc_target_at=lambda t:1200.,
        time_limit_s=6)
    assert len(rr.events)==12
    assert all(a>b for a,b in observed)
    assert len(qtimes)==4 and all(a>=b+pd.Timedelta(minutes=15) for a,b in qtimes)
    assert rr.events['rt_forecast_refreshed_utc'].nunique()==4
    assert rr.cashflow['da_usd']==80*30
    assert abs(rr.cashflow['phys_mwh']-rr.events['physical_poi_mw'].sum()/12)<1e-7
    assert rr.solve_failures==0

def test_hourly_history_only_completed():
    import pandas as pd
    from ptes_mpc.adapters import hourly_from_completed_5min
    t=pd.Timestamp('2025-07-14 12:05',tz='UTC')
    s=pd.Series(150.,index=pd.date_range('2025-07-14 08:00',periods=49,freq='5min',tz='UTC'))
    hours=hourly_from_completed_5min(s,t)
    assert list(hours.index)==[pd.Timestamp('2025-07-14 08:00',tz='UTC'),pd.Timestamp('2025-07-14 09:00',tz='UTC'),pd.Timestamp('2025-07-14 10:00',tz='UTC'),pd.Timestamp('2025-07-14 11:00',tz='UTC')]
    assert (hours.to_numpy()==150).all()
    bad=s.copy();bad.loc[t]=155
    with pytest.raises(ValueError):hourly_from_completed_5min(bad,t)
