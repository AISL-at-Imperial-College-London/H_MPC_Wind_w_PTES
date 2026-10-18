"""Targeted regression checks for October 11 LL forensic audit."""
import numpy as np
from ptes_mpc.physics import PlantParams,PlantState,execute_step
from ptes_mpc.ll_mpc import solve_ll


def test_discharge_near_empty_is_replay_feasible():
    prm=PlantParams()
    state=PlantState(20.,'discharge',0)
    sol=solve_ll([50.],[100.],state,0.,prm,
                 terminal_beta_usd_per_mwhth2=0.,solver_backend='highs',
                 time_limit_s=8.,mip_rel_gap=.0001)
    assert sol.success
    state2,info=execute_step(state,sol.modes[0],float(sol.charge_mw[0]),
                             float(sol.discharge_mw[0]),50.,prm)
    assert state2.energy_mwhth>=0
    assert info['poi_mw']<=prm.poi_limit_mw


def test_charge_near_full_is_replay_feasible():
    prm=PlantParams()
    state=PlantState(prm.capacity_mwhth-7.,'charge',0)
    sol=solve_ll([0.],[-100.],state,prm.capacity_mwhth,prm,
                 terminal_beta_usd_per_mwhth2=0.,solver_backend='highs',
                 time_limit_s=8.,mip_rel_gap=.0001)
    assert sol.success
    state2,info=execute_step(state,sol.modes[0],float(sol.charge_mw[0]),
                             float(sol.discharge_mw[0]),0.,prm)
    assert state2.energy_mwhth<=prm.capacity_mwhth


def test_inactive_roundoff_power_never_enters_thermal_polynomials():
    prm=PlantParams()
    state=PlantState(50.,'discharge',0)
    # 1e-12 MW in the inactive charge variable is numerical solver noise,
    # not an actual charge action with a thermodynamic power transfer.
    clean,clean_log=execute_step(state,'discharge',0.,100.,50.,prm)
    noisy,noisy_log=execute_step(state,'discharge',3e-12,100.,50.,prm)
    assert abs(clean.energy_mwhth-noisy.energy_mwhth)<1e-10
    assert noisy_log['charge_mw']==0.
    state=PlantState(50.,'charge',0)
    clean,clean_log=execute_step(state,'charge',60.,0.,50.,prm)
    noisy,noisy_log=execute_step(state,'charge',60.,3e-12,50.,prm)
    assert abs(clean.energy_mwhth-noisy.energy_mwhth)<1e-10
    assert noisy_log['discharge_mw']==0.
