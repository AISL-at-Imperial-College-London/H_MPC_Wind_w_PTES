"""Configuration/physics invariants independent of having a Gurobi licence."""
import numpy as np
import pytest
from unittest.mock import patch
from types import SimpleNamespace
from ptes_mpc.physics import PlantParams, PlantState, execute_step
from ptes_mpc.hl_market import solve_hl_market
from ptes_mpc.linear_mip import MIP
from ptes_mpc.ll_mpc import solve_ll


def test_hl_bid_bounds_no_more_negative_than_ptes_charge_rating():
    prm=PlantParams()
    w=np.array([[25.]*24,[75.]*24])
    p=np.array([[40.]*24,[40.]*24])
    sol=solve_hl_market(w,p,[.5,.5],500.,np.full(24,100.),np.full(24,1.),
                        prm=prm,solver_backend='highs',time_limit_s=12,mip_rel_gap=.05)
    assert sol.success,sol.message
    assert np.min(sol.commitment_mw)>=-prm.charge_max_mw-1e-6
    assert np.max(sol.commitment_mw)<=prm.poi_limit_mw+1e-6


def test_hl_gurobi_backend_is_selected_explicitly():
    # One-scenario synthetic case avoids any actual Gurobi dependency.
    synthetic=SimpleNamespace(
        x=np.zeros(1),fun=0.,success=True,status=2,message='mock Gurobi',
        mip_gap=0.,runtime_seconds=.1,best_bound=0.)
    w=np.array([[50.]*24]);p=np.array([[20.]*24])
    with patch.object(MIP,'solve_gurobi',side_effect=RuntimeError('selected Gurobi')) as mocked:
        with pytest.raises(RuntimeError,match='selected Gurobi'):
            solve_hl_market(w,p,[1.],0.,np.zeros(24),np.ones(24),
                            solver_backend='gurobi')
        assert mocked.call_count==1


def test_ll_explicit_gurobi_backend_does_not_fall_back_silently():
    with patch.object(MIP,'solve_exact_miqp',side_effect=RuntimeError('bad licence')):
        with pytest.raises(RuntimeError,match='bad licence'):
            solve_ll([50.],[30.],PlantState(200.,'off',6),200.,
                     solver_backend='gurobi')


def test_regulator_forces_off_when_only_subminimum_discharge_possible():
    prm=PlantParams()
    before=PlantState(1000.,'discharge',0)
    after,log=execute_step(before,'discharge',0,100.,prm.poi_limit_mw-20.,prm)
    assert after.mode=='off'
    assert after.off_elapsed_steps==1
    assert log['discharge_mw']==0.
    assert log['regulatory_action']=='subminimum_forced_off'
    assert log['poi_mw']<=prm.poi_limit_mw
