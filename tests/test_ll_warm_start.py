"""Warm-start partial binary mapping and LL wiring; independent of Gurobi install."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from unittest.mock import patch
from ptes_mpc.physics import PlantState,PlantParams
from ptes_mpc.warm_start import shift_ll_binary_start,capture_ll_binary_seed
from ptes_mpc.ll_mpc import solve_ll
from ptes_mpc.linear_mip import MIP
from ptes_mpc import harness


def small_plan():
    # An old plan where the first decision was 'off' (now executed),
    # followed by standby, charge, charge. All physical steps are causal.
    modes=['off','standby','charge','charge']
    names=[]
    for t in range(4):
        for mode in ('off','standby','charge','discharge'):
            names.append(f'y_{t}_{mode}')
        for edge in (('off','standby'),('standby','charge'),('charge','standby')):
            names.append(f'z_{t}_{edge[0]}_{edge[1]}')
        for family in ('charge','discharge'):
            for seg in range(5):names.append(f'k{t}_{family}_seg{seg}')
    previous={f'k{t}_charge_seg{i}':float(i==2 and modes[t]=='charge')
              for t in range(4) for i in range(5)}
    previous.update({f'k{t}_discharge_seg{i}':0. for t in range(4) for i in range(5)})
    sol=SimpleNamespace(success=True,modes=modes,warm_start_seed=previous)
    return sol,names,[1]*len(names)


def test_shifted_binary_modes_transitions_segments_are_correct():
    prev,names,ints=small_plan()
    warm,reason=shift_ll_binary_start(prev,PlantState(50.,'off',6),names,ints,6)
    m={names[i]:v for i,v in warm.items()}
    assert reason=='shifted_partial_binary_start'
    assert m['y_0_standby']==1 and m['y_1_charge']==1
    assert m['z_0_off_standby']==1 and m['z_1_standby_charge']==1
    assert m['k1_charge_seg2']==1 and m['k1_charge_seg1']==0
    assert not any(name.startswith('y_3_') for name in m)


def test_skip_after_regulatory_override_or_short_off_time():
    old,names,integers=small_plan()
    assert shift_ll_binary_start(old,PlantState(50.,'charge',0),names,integers,6)[1]=='executed_mode_differs_from_previous_plan'
    assert shift_ll_binary_start(old,PlantState(50.,'off',1),names,integers,6)[1]=='shifted_minimum_off_time_invalid'


def test_no_uncontrolled_continuous_or_wind_binary_starts():
    prev,names,integers=small_plan()
    names+=['pc_0','energy_after_0','wind_above_dispatch_cap_0']
    integers +=[0,0,1]
    warm,_=shift_ll_binary_start(prev,PlantState(50.,'off',6),names,integers,6)
    assert all(names[i] not in ('pc_0','energy_after_0','wind_above_dispatch_cap_0') for i in warm)


def test_binary_seed_filters_nonintegral_components():
    names=['y_0_off','k0_charge_seg0','pc_0','wind_above_dispatch_cap_0']
    seed=capture_ll_binary_seed(names,[1,1,0,1],np.array([1.,0.,90.,1.]))
    assert seed=={'y_0_off':1.,'k0_charge_seg0':0.}


def test_exact_ll_passes_mapped_start_to_gurobi_method():
    plan,names,ints=small_plan()
    # Need a 48-step previous plan to match the new 3-step horizon.
    plan.modes=['off','standby','charge','charge']
    seen=[]
    def fake_solve(self,*args,**kwargs):
        seen.append(kwargs.get('mip_start'))
        return SimpleNamespace(x=None,status=9,message='fake solver',success=False,mip_gap=None)
    with patch.object(MIP,'solve_exact_miqp',new=fake_solve):
        result=solve_ll([100.]*3,[40.]*3,PlantState(50.,'off',6),50.,
            solver_backend='gurobi',tracking_policy='none',
            previous_solution=plan,warm_start_enabled=True)
    assert not result.success
    assert seen and seen[0] and len(seen[0])>0
    assert all(0<=i<2000 for i in seen[0])


def test_new_simulator_passes_previous_ll_result_on_second_step(tmp_path,monkeypatch):
    start=pd.Timestamp('2025-12-18T06:00:00Z')
    hist=pd.Series(120.,index=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min'))
    future=pd.Series(120.,index=pd.date_range(start,periods=12,freq='5min'))
    rt=pd.Series(20.,index=pd.date_range(start-pd.Timedelta(minutes=15),periods=8,freq='15min'))
    hidx=pd.date_range(start,periods=8,freq='h')
    solutions=[]
    def fake_ll(wind,prices,state,reference,prm,**kwargs):
        solutions.append((kwargs.get('previous_solution'),kwargs.get('warm_start_enabled')))
        return SimpleNamespace(success=True,charge_mw=np.zeros(48),discharge_mw=np.zeros(48),
              modes=['off']*48,wind_cap_mw=None,status=2,message='mock',solver_backend='mock',
              mip_gap=0.,full_result=None,max_quadratic_approx_error_usd=0.)
    monkeypatch.setattr(harness,'solve_ll',fake_ll)
    r=harness.simulate_5min(start_utc=start,n_steps=12,
        historic_wind_5min=hist,realized_wind_5min=future,realized_rt15_price=rt,
        da_commitments_hourly=pd.Series(100.,index=hidx),da_prices_hourly=pd.Series(30.,index=hidx),
        cleared_at_utc=start-pd.Timedelta(hours=2),wind_predictor=lambda t,obs:np.full(48,120.),
        rt_predictor=lambda *a:np.full(16,20.),initial_state=PlantState(50.,'off',6),
        soc_target_at=lambda t:50.,warm_start_enabled=True)
    assert len(r.events)==12
    assert solutions[0]==(None,True)
    assert solutions[1][0] is not None and solutions[1][1] is True


def test_causal_problem_snapshot_is_replayable(tmp_path,monkeypatch):
    import json
    start=pd.Timestamp('2025-12-18T06:00:00Z')
    hist=pd.Series(120.,index=pd.date_range(start-pd.Timedelta(hours=12),periods=144,freq='5min'))
    future=pd.Series(120.,index=pd.date_range(start,periods=12,freq='5min'))
    rt=pd.Series(20.,index=pd.date_range(start-pd.Timedelta(minutes=15),periods=8,freq='15min'))
    hidx=pd.date_range(start,periods=8,freq='h')
    def fake_ll(wind,prices,state,reference,prm,**kwargs):
        return SimpleNamespace(success=True,charge_mw=np.zeros(48),discharge_mw=np.zeros(48),
            modes=['off']*48,warm_start_seed={'y_0_off':1.},wind_cap_mw=None,
            status=2,message='fake',solver_backend='fake',mip_gap=0.,full_result=None,
            max_quadratic_approx_error_usd=0.)
    monkeypatch.setattr(harness,'solve_ll',fake_ll)
    save=tmp_path/'snapshot.json'
    harness.simulate_5min(start_utc=start,n_steps=12,historic_wind_5min=hist,
        realized_wind_5min=future,realized_rt15_price=rt,
        da_commitments_hourly=pd.Series(100.,index=hidx),da_prices_hourly=pd.Series(30.,index=hidx),
        cleared_at_utc=start-pd.Timedelta(hours=2),wind_predictor=lambda t,obs:np.full(48,120.),
        rt_predictor=lambda *a:np.full(16,20.),initial_state=PlantState(50.,'off',6),
        soc_target_at=lambda t:50.,warm_start_enabled=True,
        snapshot_at_utc=start+pd.Timedelta(minutes=5),snapshot_path=save)
    d=json.loads(save.read_text())
    assert d['timestamp_utc']==(start+pd.Timedelta(minutes=5)).isoformat()
    assert d['previous_modes']==['off']*48
    assert d['previous_warm_start_seed']=={'y_0_off':1.}
    assert len(d['wind_forecast_mw'])==48
    assert 'realized_wind' not in d
