"""Fast deterministic baseline QA (no Gurobi; no scored-week runs)."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from deterministic_hl_baseline import mean_hl_inputs,ensemble_sha256

def test_math():
    rng=np.random.default_rng(37)
    w=rng.uniform(0,230,size=(50,84))
    p=rng.normal(30,80,size=(50,84))
    pi=rng.dirichlet(np.arange(1,51))
    a,b,c=mean_hl_inputs(w,p,pi)
    assert a.shape == b.shape == (1,84)
    assert c.shape == (1,) and c[0]==1.
    assert np.allclose(a[0],sum(pi[i]*w[i] for i in range(50)),atol=1e-12)
    assert np.allclose(b[0],sum(pi[i]*p[i] for i in range(50)),atol=1e-12)
    assert np.isclose(a[0,0],float(np.dot(pi,w[:,0])))
    assert np.isclose(b[0,10],float(np.dot(pi,p[:,10])))
    assert ensemble_sha256(w,p,pi)==ensemble_sha256(w.copy(),p.copy(),pi.copy())
    try:mean_hl_inputs(w[:-1],p[:-1],pi[:-1])
    except AssertionError:pass
    else:raise AssertionError('Failure to detect incorrect count')
    try:mean_hl_inputs(w,p,pi[::-1]*2)
    except AssertionError:pass
    else:raise AssertionError('Failure to detect incorrect probability mass')
    print('PASS: weighted 50-to-1 mathematical transformation, shape/weights checks, checksum stability')


def test_solver_interfaces():
    # No modification of optimization logic; a one-scenario wind-only and PTES
    # HL problem is supported by both existing solvers.
    from wind_only.hl import solve_wind_hl
    from ptes_mpc.hl_market import solve_hl_market
    from ptes_mpc.physics import PlantParams
    from numpy import full
    wind=full((1,84),120.);da=full((1,84),50.); weights=np.array([1.])
    spread=full(84,-1.);risk=full(84,4.)
    r=solve_wind_hl(wind,da,weights,spread,risk,solver_backend='highs',time_limit_s=30)
    assert r.success and r.commitment_mw.shape==(24,),r.message
    print('PASS: original wind-only HL LP with one weighted-mean trajectory')
    r2=solve_hl_market(wind,da,weights,200.,spread,risk,
       solver_backend='highs',time_limit_s=30,mip_rel_gap=.05,hl_curtailment_enabled=True,
       prm=PlantParams())
    assert r2.success and r2.commitment_mw.shape==(24,) and r2.reference_mwhth.shape==(84,),r2.message
    print('PASS: original PTES HL MILP with one weighted-mean trajectory (HiGHS smoke)')

if __name__=='__main__':
    test_math()
    test_solver_interfaces()
    print('DETERMINISTIC V3 BASELINE CHECKS PASSED (Gurobi full campaign is run locally in Spyder).')
