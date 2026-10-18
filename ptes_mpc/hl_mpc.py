"""LEGACY HL benchmark only: use hl_market.py for the study (ERCOT bid/settlement)."""
"""Stochastic hourly HL scheduling MILP. Development *prototype*.

Important: the objective is expected physical DA-price value LESS an explicitly
labelled scheduling-deviation risk proxy, NOT actual ERCOT RT settlement. For
published economic results, replace the proxy with an empirically validated
conditional RT settlement model; record this assumption in every output.
"""
from dataclasses import dataclass
import numpy as np
from .linear_mip import MIP
from .physics import PlantParams
from .pwa import add_mapping

@dataclass
class HLSolution:
    success: bool
    message: str
    commitment_mw: np.ndarray|None
    reference_mwhth: np.ndarray|None
    objective_proxy_usd: float|None
    mip_gap: float|None
    full_result: object=None

def solve_hl(wind_scenarios,price_scenarios,weights,initial_energy_mwhth,
             prm:PlantParams=PlantParams(), deviation_risk_usd_mwh=8.,
             terminal_value_usd_mwhth=0.,time_limit_s=90.,mip_rel_gap=.04):
    w=np.asarray(wind_scenarios,dtype=float); p=np.asarray(price_scenarios,dtype=float)
    pi=np.asarray(weights,dtype=float).ravel()
    if w.shape!=p.shape or w.ndim!=2 or len(pi)!=len(w) or w.shape[1]<24:
        raise ValueError('Require (scenario, hourly horizon >=24) wind and DA prices')
    ns,nh=w.shape
    if not np.isfinite(w).all() or not np.isfinite(p).all() or not np.isfinite(pi).all():
        raise ValueError('non-finite scenario inputs')
    if (w<0).any() or (w>prm.rated_wind_mw).any() or (pi<=0).any() or abs(pi.sum()-1)>1e-8:
        raise ValueError('invalid wind scenario or weights')
    if not 0<=initial_energy_mwhth<=prm.capacity_mwhth:raise ValueError('initial SoC')
    # q[h] shared across the first 24h; the 60h buffer uses scenario-specific q.
    m=MIP();q_first={};q_tail={}; pc={};pd={};qc={};qd={}; E={}; yc={};yd={}; dev_plus={};dev_minus={}
    retain=prm.daily_retention**(1/24)
    for h in range(24): q_first[h]=m.v(f'q_DA_{h}',-prm.charge_max_mw,prm.poi_limit_mw)
    for s in range(ns):
        for h in range(nh):
            q=q_first[h] if h<24 else m.v(f'q_tail_{s}_{h}',-prm.charge_max_mw,prm.poi_limit_mw)
            if h>=24:q_tail[s,h]=q
            yc[s,h]=m.v(f'charge_on_{s}_{h}',0,1,True)
            yd[s,h]=m.v(f'discharge_on_{s}_{h}',0,1,True)
            m.c({yc[s,h]:1,yd[s,h]:1},hi=1)
            pc[s,h]=m.v(f'charge_{s}_{h}',0,prm.charge_max_mw)
            pd[s,h]=m.v(f'discharge_{s}_{h}',0,prm.discharge_max_mw)
            qc[s,h]=m.v(f'charge_thermal_{s}_{h}',0,prm.charge_max_mw*3)
            qd[s,h]=m.v(f'discharge_thermal_{s}_{h}',0,prm.discharge_max_mw*4)
            E[s,h]=m.v(f'E_{s}_{h}',0,prm.capacity_mwhth,
                       obj=-pi[s]*terminal_value_usd_mwhth if h==nh-1 else 0)
            m.c({pc[s,h]:1,yc[s,h]:-prm.charge_min_mw},lo=0)
            m.c({pc[s,h]:1,yc[s,h]:-prm.charge_max_mw},hi=0)
            m.c({pd[s,h]:1,yd[s,h]:-prm.discharge_min_mw},lo=0)
            m.c({pd[s,h]:1,yd[s,h]:-prm.discharge_max_mw},hi=0)
            add_mapping(m,pc[s,h],qc[s,h],yc[s,h],'charge',prm,f's{s}_h{h}')
            add_mapping(m,pd[s,h],qd[s,h],yd[s,h],'discharge',prm,f's{s}_h{h}')
            m.c({pc[s,h]:-1,pd[s,h]:1},lo=-prm.poi_limit_mw-w[s,h],hi=prm.poi_limit_mw-w[s,h])
            co={E[s,h]:1,qc[s,h]:-1,qd[s,h]:1}
            if h:co[E[s,h-1]]=-retain;rhs=0
            else:rhs=retain*initial_energy_mwhth
            m.c(co,lo=rhs,hi=rhs)
            # g - q = w - pc + pd - q. Deviation split >= |g-q| at optimum.
            dev_plus[s,h]=m.v(f'dev_plus_{s}_{h}',0,2*prm.poi_limit_mw,
                               obj=pi[s]*deviation_risk_usd_mwh)
            dev_minus[s,h]=m.v(f'dev_minus_{s}_{h}',0,2*prm.poi_limit_mw,
                                obj=pi[s]*deviation_risk_usd_mwh)
            m.c({q:1,pc[s,h]:1,pd[s,h]:-1,dev_plus[s,h]:1,dev_minus[s,h]:-1},
                lo=w[s,h],hi=w[s,h])
            # Expected physical day-ahead value: price*(w - pc + pd).
            # The contribution of w is constant and is added after solve.
            m.obj[pc[s,h]]+=pi[s]*p[s,h]
            m.obj[pd[s,h]]-=pi[s]*p[s,h]
    res=m.solve(time_limit_s,mip_rel_gap)
    if res.x is None:return HLSolution(False,str(res.message),None,None,None,None,res)
    q=np.array([res.x[q_first[h]] for h in range(24)])
    ref=np.array([sum(pi[s]*res.x[E[s,h]] for s in range(ns)) for h in range(min(24,nh))])
    physical_constant=float(np.sum(pi[:,None]*p*w))
    return HLSolution(bool(res.success),str(res.message),q,ref,
                      float(physical_constant-res.fun),getattr(res,'mip_gap',None),res)
