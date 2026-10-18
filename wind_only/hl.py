"""Wind-only stochastic HL day-ahead offer LP. No storage, no operating state.

Only the first 24 forecast hours are coupled to DA decisions. The generator
supplies 84h forecasts to stay identical to the PTES study; the 60h tail has
no physical state through which it could affect wind-only bids. Day-ahead
bids are common across scenarios, while physical wind dispatch is scenario
recourse, 0 <= dispatch <= available wind. No speculative DA sale above the
scenario-weighted mean available wind (configurable conservative study cap).

Objective = expected DA+RT cash flow minus an optional historic-IQR-derived
imbalance-risk *planning preference*. The risk proxy is NOT a settlement fee.
"""
from dataclasses import dataclass
import numpy as np
from .linear_mip import MIP

@dataclass
class WindHLResult:
    success:bool
    message:str
    commitment_mw:np.ndarray|None
    expected_dispatch_mw:np.ndarray|None
    expected_wind_mw:np.ndarray|None
    objective_proxy_usd:float|None
    mip_gap:float|None
    runtime_seconds:float|None
    model_nvars:int
    model_nrows:int


def solve_wind_hl(wind_scenarios,da_scenarios,weights,spread_hourly,risk_hourly,
                  *,rated_mw=230.,poi_limit_mw=230.,risk_scale=1.,
                  offer_cap_fraction=1.,solver_backend='highs',time_limit_s=60.,
                  mip_rel_gap=.02):
    w=np.asarray(wind_scenarios,float);p=np.asarray(da_scenarios,float)
    probs=np.asarray(weights,float).ravel();sp=np.asarray(spread_hourly,float).ravel()
    risk=np.asarray(risk_hourly,float).ravel()
    if w.ndim!=2 or w.shape!=p.shape or w.shape[1]<24 or w.shape[0]!=len(probs):
        raise ValueError('Expected aligned (n_scenarios,84_hours) joint forecast')
    if not (len(sp)>=24 and len(risk)>=24 and abs(probs.sum()-1)<1e-7 and (probs>0).all()):
        raise ValueError('Invalid scenario weights or spread/risk forecast')
    if not all(np.isfinite(a).all() for a in (w,p,probs,sp,risk)):
        raise ValueError('Nonfinite forecast')
    if (w<0).any() or (w>rated_mw+1e-6).any() or (risk<0).any() or risk_scale<0:
        raise ValueError('Invalid wind/risk')
    if not (0<offer_cap_fraction<=1):raise ValueError('offer_cap_fraction must be (0,1]')
    n=len(probs);m=MIP();q={};dispatch={}
    mean_w=np.dot(probs,w[:,:24]);cap=np.minimum(poi_limit_mw,offer_cap_fraction*mean_w)
    for h in range(24):
        q[h]=m.v(f'DA_bid_mw_h{h}',0.,float(cap[h]))
        for s in range(n):
            # Forecast wind above physical POI is not a second source of generation.
            g=dispatch[s,h]=m.v(f'wind_dispatched_mw_s{s}_h{h}',0.,float(min(w[s,h],poi_limit_mw)),
                                 obj=-float(probs[s])*(float(p[s,h])+float(sp[h])))
            plus=m.v(f'excess_energy_mwh_s{s}_h{h}',0.,poi_limit_mw,
                     obj=float(probs[s])*risk_scale*float(risk[h]))
            minus=m.v(f'short_energy_mwh_s{s}_h{h}',0.,poi_limit_mw,
                      obj=float(probs[s])*risk_scale*float(risk[h]))
            # excess-short = dispatched - signed DA bid; each h is one hour.
            m.c({plus:1,minus:-1,g:-1,q[h]:1},lo=0.,hi=0.)
            # DA*q+RT*(g-q), expressed in minimization form.
            m.obj[q[h]]+=float(probs[s])*(float(p[s,h])+float(sp[h])-float(p[s,h]))
    if solver_backend=='highs':result=m.solve(time_limit_s,mip_rel_gap)
    elif solver_backend=='gurobi':result=m.solve_gurobi(time_limit_s,mip_rel_gap)
    else:raise ValueError('solver_backend must be highs or gurobi')
    if result.x is None or not result.success:
        return WindHLResult(False,str(result.message),None,None,mean_w,None,
                    getattr(result,'mip_gap',None),getattr(result,'runtime_seconds',None),len(m.names),len(m.rows))
    x=result.x
    bids=np.asarray([x[q[h]] for h in range(24)],float)
    exp_dispatch=np.asarray([sum(probs[s]*x[dispatch[s,h]] for s in range(n)) for h in range(24)])
    return WindHLResult(True,str(result.message),bids,exp_dispatch,mean_w,
                        float(-result.fun),getattr(result,'mip_gap',None),
                        getattr(result,'runtime_seconds',None),len(m.names),len(m.rows))
