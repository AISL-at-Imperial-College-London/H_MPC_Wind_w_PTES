"""Regression for a feasible time-limited LL incumbent (Gurobi status 9)."""
from types import SimpleNamespace
import numpy as np
import pytest
from unittest.mock import patch
from ptes_mpc.linear_mip import MIP, assess_gurobi_incumbent
from ptes_mpc.ll_mpc import solve_ll
from ptes_mpc.physics import PlantState


def test_status9_2_2637_percent_accepted_under_explicit_five_percent_cap():
    accepted, certified, relaxed = assess_gurobi_incumbent(
        status=9,solution_count=10,gap=0.022637096851347332,
        requested_gap=.02,maximum_time_limit_gap=.05,max_violation=1e-7)
    assert (accepted, certified, relaxed)==(True,False,True)


def test_status9_strict_without_cap_and_excessive_gap_still_rejected():
    assert assess_gurobi_incumbent(status=9,solution_count=10,gap=.022637,
        requested_gap=.02,max_violation=0.)==(False,False,False)
    assert assess_gurobi_incumbent(status=9,solution_count=10,gap=.11,
        requested_gap=.02,maximum_time_limit_gap=.05,max_violation=0.)==(False,False,False)
    assert assess_gurobi_incumbent(status=9,solution_count=0,gap=None,
        requested_gap=.02,maximum_time_limit_gap=.05,max_violation=0.)==(False,False,False)
    assert assess_gurobi_incumbent(status=9,solution_count=10,gap=.024,
        requested_gap=.02,maximum_time_limit_gap=.05,max_violation=0.002)==(False,False,False)


def test_requested_target_hit_is_certified_and_not_relaxed():
    assert assess_gurobi_incumbent(status=9,solution_count=10,gap=.0199,
        requested_gap=.02,maximum_time_limit_gap=.05,max_violation=0.)==(True,True,False)
    assert assess_gurobi_incumbent(status=2,solution_count=1,gap=0.,
        requested_gap=.02,maximum_time_limit_gap=.05,max_violation=0.)==(True,True,False)


def test_ll_passes_policy_only_to_its_exact_miqp(monkeypatch):
    recorded=[]
    def fake_solve(self,quadratic_terms,time_limit_s,mip_rel_gap,maximum_time_limit_gap=None):
        recorded.append((time_limit_s,mip_rel_gap,maximum_time_limit_gap))
        # Validity of the Gurobi model/solution is not established in this fake;
        # test only that the policy reaches the LL backend.
        return SimpleNamespace(x=None,status=9,message='no-incumbent test',success=False,
                               mip_gap=None)
    monkeypatch.setattr(MIP,'solve_exact_miqp',fake_solve)
    sol=solve_ll([50.],[20.],PlantState(0.,'off',6),0.,
        solver_backend='gurobi',maximum_time_limit_gap=.05)
    assert not sol.success
    assert recorded==[(30.,.02,.05)]
