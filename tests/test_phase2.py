"""Agreed regulatory and terminal-episode settings."""
import numpy as np
import pandas as pd
import pytest
from ptes_mpc.physics import PlantParams,PlantState
from ptes_mpc.ll_mpc import solve_ll
from ptes_mpc.harness import simulate_5min
from ptes_mpc.episodes import SOC_CASES,run_four_soc_cases

@pytest.fixture
def onehour():
    start=pd.Timestamp('2025-07-14 10:00',tz='UTC')
    pastidx=pd.date_range(end=start-pd.Timedelta(minutes=5),periods=49,freq='5min',tz='UTC')
    realidx=pd.date_range(start,periods=12,freq='5min',tz='UTC')
    rtidx=pd.date_range(end=start-pd.Timedelta(minutes=15),periods=16,freq='15min',tz='UTC')
    rtidx=rtidx.union(pd.date_range(start,periods=4,freq='15min',tz='UTC'))
    daidx=pd.date_range(start.floor('h'),periods=6,freq='h',tz='UTC')
    return dict(start_utc=start,n_steps=12,
        historic_wind_5min=pd.Series(115.,index=pastidx),
        realized_wind_5min=pd.Series(115.,index=realidx),
        realized_rt15_price=pd.Series(40.,index=rtidx),
        da_commitments_hourly=pd.Series(100.,index=daidx),
        da_prices_hourly=pd.Series(25.,index=daidx),
        cleared_at_utc=start-pd.Timedelta(hours=5),
        wind_predictor=lambda issue,history:np.full(48,float(history.iloc[-1])),
        rt_predictor=lambda issue,old,da,wind:np.full(16,40.),
        soc_target_at=lambda time:1300.,
        time_limit_s=8,mip_rel_gap=.1,quadratic_tangents=101)

def test_quadratic_epigraph_accuracy_and_not_abs_cost():
    from ptes_mpc.linear_mip import MIP,add_quadratic_epigraph
    p=PlantParams();m=MIP()
    E=m.v('E',0,p.capacity_mwhth)
    # Force a 10% gap; evaluate epigraph exact-vs-bounded approximation.
    m.c({E:1},lo=.65*p.capacity_mwhth,hi=.65*p.capacity_mwhth)
    bound=add_quadratic_epigraph(m,E,.55*p.capacity_mwhth,1.,p.capacity_mwhth,101)
    result=m.solve(5,.01)
    true=(.1*p.capacity_mwhth)**2
    assert result.x is not None
    assert 0<=true-result.fun<=bound+1e-4

def test_terminal_tracking_changes_decision():
    p=PlantParams()
    w=np.full(12,100.)
    prices=np.full(12,100.)
    initial=PlantState(1000.,'off',6)
    out=solve_ll(w,prices,initial,1300.,p,terminal_beta_usd_per_mwhth2=1.,
                 quadratic_tangents=101,time_limit_s=10,mip_rel_gap=.08,solver_backend='highs')
    assert out.charge_mw is not None,out.message
    assert out.solver_backend=='highs_quadratic_epigraph_milp'
    assert len(out.charge_mw)==12
    assert out.objective_usd is not None

def test_4_soc_cases_smoke(onehour,tmp_path):
    p=PlantParams()
    out=run_four_soc_cases(base_kwargs=onehour,prm=p,output_dir=tmp_path,
         terminal_beta_usd_per_mwhth2=1.,
         hl_schedule_factory=lambda case,state: {'da_commitments_hourly': onehour['da_commitments_hourly'],
                                                 'soc_target_at': lambda time: state.energy_mwhth})
    assert list(out)==[c.name for c in SOC_CASES]
    for name,result in out.items():
        assert len(result.events)==12
        assert result.solve_failures==0
        assert result.cashflow['total_usd']==pytest.approx(result.cashflow['da_usd']+result.cashflow['rt_usd'])
        if name.endswith('free'):
            assert result.events['ll_horizon_steps'].eq(48).all()
        else:
            assert result.events['ll_horizon_steps'].iloc[0]==12
            assert result.events['ll_horizon_steps'].iloc[-1]==1
        assert result.events['physical_poi_mw'].between(-230.00001,230.00001).all()
        assert not result.events['safety_override'].any()
        if name.startswith('empty'):
            assert result.initial_energy_mwhth==0
        else:
            assert result.initial_energy_mwhth==p.capacity_mwhth
        if name.endswith('free'):
            assert result.actual_terminal_error_mwhth is None
            assert result.terminal_soft_penalty_usd==0
        else:
            assert result.actual_terminal_error_mwhth is not None
            assert result.terminal_soft_penalty_usd==pytest.approx(result.actual_terminal_error_mwhth**2)
    csv=pd.read_csv(tmp_path/'case_summary.csv')
    assert len(csv)==4
    assert csv['total_usd'].notna().all()

def test_revenue_not_reduced_by_terminal_penalty(onehour):
    p=PlantParams()
    out=simulate_5min(**onehour,initial_state=PlantState(0,'off',6),
         episode_terminal_target_mwhth=p.capacity_mwhth)
    assert out.actual_terminal_error_mwhth is not None
    assert out.terminal_soft_penalty_usd>0
    assert out.cashflow['total_usd']==pytest.approx(out.cashflow['da_usd']+out.cashflow['rt_usd'])

def test_clipping_propagates_actual_mode_soc_and_ledger(onehour):
    p=PlantParams()
    setup=onehour.copy()
    setup['realized_wind_5min']=pd.Series(220.,index=setup['realized_wind_5min'].index)
    setup['wind_predictor']=lambda issue,history:np.full(48,100.)
    setup['rt_predictor']=lambda issue,old,da,wind:np.full(16,200.)
    setup['soc_target_at']=lambda t:0.
    output=simulate_5min(**setup,initial_state=PlantState(1000.,'discharge',0),
                           episode_terminal_target_mwhth=0,
                           terminal_beta_usd_per_mwhth2=1.)
    first=output.events.iloc[0]
    assert first['commanded_mode']=='discharge'
    assert first['actual_mode']=='off'
    assert first['regulatory_action']=='subminimum_forced_off'
    assert first['discharge_mw']==0
    assert first['physical_poi_mw']==220
    assert first['off_elapsed_steps']==1
    assert output.regulatory_interventions>=1
    assert output.forced_off_interventions>=1
    assert output.cashflow['phys_mwh']==pytest.approx(output.events['physical_poi_mw'].sum()/12)
    assert output.final_state.energy_mwhth==pytest.approx(output.events['tank_mwhth'].iloc[-1])
