"""Manuscript-style stochastic HL MILP with explicit DA/RT deviation economics.

Economic proxy: prospective RT15 prices are NOT predicted by the short-horizon RT
XGBoost. HL uses only pre-bid historical RT-DA spreads, conditioned on local
clock hour, to set E[RT|DA scenario] and an empirically-calibrated symmetric
imbalance-risk *utility* adjustment. This is a strategy assumption, not actual
ERCOT settlement: actual signed settlement uses one observed RT SPP ex post.

Physical POI is constrained independently of commercial commitments.
The base model retains simple HL charge/discharge mode binaries, original
weighted-mean SoC reference, and equal state at end of first 24h.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .linear_mip import MIP
from .physics import PlantParams
from .pwa import add_mapping

@dataclass
class HLMarketSolution:
    success:bool
    message:str
    commitment_mw:np.ndarray|None
    reference_mwhth:np.ndarray|None
    objective_proxy_usd:float|None
    mip_gap:float|None
    estimated_rt_spread_usd_mwh:np.ndarray|None
    risk_premium_usd_mwh:np.ndarray|None
    scenario_end_day1_mwhth:np.ndarray|None=None
    full_result:object=None
    solver_backend:str="unknown"
    runtime_seconds:float|None=None
    best_bound:float|None=None
    model_nvars:int|None=None
    model_nrows:int|None=None
    expected_wind_dispatch_mw:np.ndarray|None=None
    wind_curtailment_enabled:bool=False


def estimate_spread_from_past(*,rt15,da_hourly,decision_utc,delivery_times_utc,
                               window_days=28):
    """Return as-of historic mean RT-DA spread and half IQR per local hour.

    Half-IQR is a predeclared proxy for imbalance exposure, NOT a settlement
    rule or deducted from realised cashflows. Never read prices >= decision.
    """
    cutoff=pd.Timestamp(decision_utc).tz_convert('UTC')
    a=pd.Series(rt15,dtype=float).sort_index();d=pd.Series(da_hourly,dtype=float).sort_index()
    a.index=pd.to_datetime(a.index,utc=True);d.index=pd.to_datetime(d.index,utc=True)
    a=a[(a.index+pd.Timedelta(minutes=15)<=cutoff)&(a.index>=cutoff-pd.Timedelta(days=window_days))]
    if a.empty:raise ValueError('no eligible completed historical RT data')
    dh=d.reindex(a.index.floor('h')).to_numpy(float)
    spread=a.to_numpy(float)-dh
    hours=a.index.tz_convert('America/Chicago').hour
    filt=np.isfinite(spread)
    if filt.sum()<4*24*5:raise ValueError('insufficient historical RT-DA spread history')
    means={};risks={}
    for h in range(24):
        x=spread[filt&(hours==h)]
        if len(x)<8:raise ValueError(f'not enough local-hour historic RT spread for {h}')
        means[h]=float(np.mean(x))
        risks[h]=float(max(0.01,(np.quantile(x,.75)-np.quantile(x,.25))/2))
    target_hours=pd.DatetimeIndex(delivery_times_utc).tz_convert('America/Chicago').hour
    return np.asarray([means[h] for h in target_hours]),np.asarray([risks[h] for h in target_hours])


def solve_hl_market(wind_scenarios,da_scenarios,weights,initial_energy_mwhth,
                    rt_spread_hourly,risk_premium_hourly,*,prm=PlantParams(),
                    time_limit_s=30.,mip_rel_gap=.05,solver_backend="highs",
                    gurobi_threads=0, gurobi_seed=1, gurobi_output=False,
                    hl_curtailment_enabled=False,hl_risk_scale=1.):
    """24 common DA commitments + scenario recourse over >=24 hourly steps.

    Reward within first 24h = DA*q + E[RT]*(g-q) - risk*|g-q|.
    For buffer hours the commitment is not being decided at this issue; use
    prospective DA value of physical electricity for recourse/SoC planning.
    No discretionary wind curtailment in the Horse Creek base case.
    """
    w=np.asarray(wind_scenarios,float);p=np.asarray(da_scenarios,float)
    pi=np.asarray(weights,float).ravel(); spread=np.asarray(rt_spread_hourly,float).ravel()
    risk=np.asarray(risk_premium_hourly,float).ravel()
    if w.ndim!=2 or w.shape!=p.shape or len(pi)!=len(w) or w.shape[1]<24:
        raise ValueError('expected (scenarios, >=24h) wind and price')
    ns,nh=w.shape
    if len(spread)!=nh or len(risk)!=nh or min(pi)<=0 or abs(pi.sum()-1)>1e-8:
        raise ValueError('incorrect scenario weights, RT spread or risk arrays')
    if not np.isfinite(w).all() or not np.isfinite(p).all() or not np.isfinite(spread).all() or not np.isfinite(risk).all():
        raise ValueError('non-finite inputs')
    if np.any(w<0) or np.any(w>prm.rated_wind_mw) or np.any(risk<=0):
        raise ValueError('wind or risk outside valid range')
    if not np.isfinite(hl_risk_scale) or hl_risk_scale<0:
        raise ValueError('HL risk scale must be nonnegative and finite')
    if not 0<=initial_energy_mwhth<=prm.capacity_mwhth:
        raise ValueError('invalid initial energy')
    m=MIP();q=[m.v(f'q_DA_{h}',-prm.charge_max_mw,prm.poi_limit_mw) for h in range(24)]
    E={};wind_dispatch={};retain=prm.daily_retention**(1/24)
    constant=0.
    for s in range(ns):
        for h in range(nh):
            pc=m.v(f'charge_{s}_{h}',0,prm.charge_max_mw)
            pd=m.v(f'discharge_{s}_{h}',0,prm.discharge_max_mw)
            qc=m.v(f'charge_th_{s}_{h}',0,prm.charge_max_mw*3)
            qd=m.v(f'discharge_th_{s}_{h}',0,prm.discharge_max_mw*4)
            yc=m.v(f'yc_{s}_{h}',0,1,True);yd=m.v(f'yd_{s}_{h}',0,1,True)
            E[s,h]=m.v(f'E_{s}_{h}',0,prm.capacity_mwhth)
            m.c({yc:1,yd:1},hi=1)
            m.c({pc:1,yc:-prm.charge_min_mw},lo=0)
            m.c({pc:1,yc:-prm.charge_max_mw},hi=0)
            m.c({pd:1,yd:-prm.discharge_min_mw},lo=0)
            m.c({pd:1,yd:-prm.discharge_max_mw},hi=0)
            add_mapping(m,pc,qc,yc,'charge',prm,f's{s}_h{h}')
            add_mapping(m,pd,qd,yd,'discharge',prm,f's{s}_h{h}')
            if hl_curtailment_enabled:
                wd=wind_dispatch[s,h]=m.v(f'wind_dispatched_{s}_{h}',0.,float(w[s,h]))
                # Hourly HL recourse can choose physically available wind; 5-minute
                # ramp and realization are enforced separately by the LL/replay.
                m.c({wd:1,pc:-1,pd:1},lo=-prm.poi_limit_mw,hi=prm.poi_limit_mw)
            else:
                wd=None
                m.c({pc:-1,pd:1},lo=-prm.poi_limit_mw-w[s,h],hi=prm.poi_limit_mw-w[s,h])
            co={E[s,h]:1,qc:-1,qd:1}
            if h:co[E[s,h-1]]=-retain;rhs=0.
            else:rhs=retain*initial_energy_mwhth
            m.c(co,lo=rhs,hi=rhs)
            if h<24:
                # physical POI g = wind - pc + pd; g-q = long-short.
                long=m.v(f'long_{s}_{h}',0,2*prm.poi_limit_mw,
                         obj=pi[s]*(hl_risk_scale*risk[h]-(p[s,h]+spread[h])))
                short=m.v(f'short_{s}_{h}',0,2*prm.poi_limit_mw,
                          obj=pi[s]*(hl_risk_scale*risk[h]+(p[s,h]+spread[h])))
                balance={q[h]:1,pc:1,pd:-1,long:1,short:-1}
                if hl_curtailment_enabled:
                    balance[wd]=-1
                    m.c(balance,lo=0.,hi=0.)
                else:
                    m.c(balance,lo=w[s,h],hi=w[s,h])
                m.obj[q[h]]-=pi[s]*p[s,h]
            else:
                m.obj[pc]+=pi[s]*p[s,h]
                m.obj[pd]-=pi[s]*p[s,h]
                if hl_curtailment_enabled:
                    m.obj[wd]-=pi[s]*p[s,h]
                else:
                    constant+=pi[s]*p[s,h]*w[s,h]
    # Manuscript Eq (23): shared end-of-first-day storage energy across scenarios.
    for s in range(1,ns):
        m.c({E[s,23]:1,E[0,23]:-1},lo=0,hi=0)
    if solver_backend=="gurobi":
        r=m.solve_gurobi(time_limit_s,mip_rel_gap,threads=gurobi_threads,
                         seed=gurobi_seed,gurobi_output=gurobi_output)
    elif solver_backend=="highs":
        r=m.solve(time_limit_s,mip_rel_gap)
    else:
        raise ValueError("HL solver_backend must be gurobi or highs")
    if r.x is None:
        return HLMarketSolution(False,str(r.message),None,None,None,None,spread,risk,None,r,
                                solver_backend,getattr(r,"runtime_seconds",None),
                                getattr(r,"best_bound",None),len(m.names),len(m.rows),
                                None,bool(hl_curtailment_enabled))
    ref=np.asarray([sum(pi[s]*r.x[E[s,h]] for s in range(ns)) for h in range(nh)])
    expected_wind_dispatch=np.asarray([
        sum(pi[s]*(r.x[wind_dispatch[s,h]] if hl_curtailment_enabled else w[s,h])
            for s in range(ns)) for h in range(nh)],float)
    return HLMarketSolution(bool(r.success),str(r.message),
         np.asarray([r.x[v] for v in q]),ref,float(constant-r.fun),
         getattr(r,'mip_gap',None),spread,risk,
         np.asarray([r.x[E[s,23]] for s in range(ns)]),r,
         solver_backend,getattr(r,"runtime_seconds",None),
         getattr(r,"best_bound",None),len(m.names),len(m.rows),
         expected_wind_dispatch,bool(hl_curtailment_enabled))
