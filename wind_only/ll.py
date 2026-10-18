"""Wind-only curtailment LL MILP with exact forecast min(wind, cap) and ramp.

No storage variables, PTES mode binaries, energy inventory or terminal value.
The offered DA schedule only enters as a configurable *operational preference*,
not an invented ERCOT financial penalty. DA/RT financial settlement is separate.
"""
from dataclasses import dataclass
import numpy as np
from .linear_mip import MIP

@dataclass
class WindLLResult:
    success:bool
    message:str
    cap_mw:np.ndarray|None
    predicted_dispatch_mw:np.ndarray|None
    predicted_curtail_mw:np.ndarray|None
    mip_gap:float|None
    runtime_seconds:float|None
    tracking_forecast_abs_mwh:float|None


def solve_wind_ll(wind_forecast,rt_price_forecast,prev_dispatched_mw,
                 da_commitments_5min,*,already_delivered_quarter_mwh=0.,quarter_offset=0,
                 tracking_weight_usd_per_mwh=5.,ramp_mw_per_5min=40.,rated_mw=230.,
                 solver_backend='highs',time_limit_s=30.,mip_rel_gap=.025):
    w=np.asarray(wind_forecast,float).ravel();price=np.asarray(rt_price_forecast,float).ravel()
    da=np.asarray(da_commitments_5min,float).ravel();N=len(w)
    if not (1<=N<=48 and price.shape==(N,) and da.shape==(N,)):
        raise ValueError('Expected up to 48 aligned wind, RT and DA values')
    if not (0<=prev_dispatched_mw<=rated_mw and 0<=tracking_weight_usd_per_mwh and
            0<ramp_mw_per_5min<=rated_mw and quarter_offset in (0,1,2)):
        raise ValueError('Invalid preceding dispatch/ramp/tracking')
    if not np.isfinite(w).all() or not np.isfinite(price).all() or (w<0).any() or (w>rated_mw+1e-6).any():
        raise ValueError('Bad wind or price forecast')
    m=MIP();cap={};g={};dt=1/12
    for j in range(N):
        cap[j]=m.v(f'wind_cap_{j}',0,rated_mw)
        g[j]=m.v(f'dispatched_wind_{j}',0,float(w[j]),obj=-dt*float(price[j]))
        above=m.v(f'wind_cap_below_avail_{j}',0,1,True)
        M=float(rated_mw)
        # g = min(cap, forecast available wind) exactly.
        m.c({g[j]:1,cap[j]:-1},hi=0.)
        m.c({g[j]:1,cap[j]:-1,above:M},lo=0.)
        m.c({g[j]:1,above:-M},lo=float(w[j])-M)
        # The command cap, not unrealized weather, ramps +/-40MW per 5min.
        if j:
            m.c({cap[j]:1,g[j-1]:-1},lo=-ramp_mw_per_5min,hi=ramp_mw_per_5min)
        else:
            m.c({cap[j]:1},lo=max(0.,prev_dispatched_mw-ramp_mw_per_5min),
                              hi=min(rated_mw,prev_dispatched_mw+ramp_mw_per_5min))
    tracker=[]
    if tracking_weight_usd_per_mwh>0:
        buckets={}
        for j in range(N):buckets.setdefault((quarter_offset+j)//3,[]).append(j)
        for k,ix in buckets.items():
            needed=3-quarter_offset if k==0 else 3
            if len(ix)!=needed:continue
            if not np.isfinite(da[ix]).all():continue # do not anticipate unaccepted bids
            if np.ptp(da[ix])>1e-7:raise ValueError('DA commitment varies within quarter')
            plus=m.v(f'quarter_long_{k}',0,float('inf'))
            minus=m.v(f'quarter_short_{k}',0,float('inf'))
            terms={plus:1,minus:-1}
            for j in ix:terms[g[j]]=-dt
            already=float(already_delivered_quarter_mwh) if k==0 else 0.
            m.c(terms,lo=already-float(da[ix][0])/4,hi=already-float(da[ix][0])/4)
            weight=4. if k==0 else 2. if k<=3 else 1.
            m.obj[plus]+=tracking_weight_usd_per_mwh*weight
            m.obj[minus]+=tracking_weight_usd_per_mwh*weight
            tracker.append((plus,minus))
    if solver_backend=='highs':result=m.solve(time_limit_s,mip_rel_gap)
    elif solver_backend=='gurobi':result=m.solve_gurobi(time_limit_s,mip_rel_gap)
    else:raise ValueError('solver_backend must be highs or gurobi')
    if result.x is None or not result.success:
        return WindLLResult(False,str(result.message),None,None,None,getattr(result,'mip_gap',None),
                            getattr(result,'runtime_seconds',None),None)
    x=result.x
    caps=np.asarray([x[cap[j]] for j in range(N)],float)
    dispatched=np.minimum(w,caps)
    pred_abs=float(sum(x[plus]+x[minus] for plus,minus in tracker))
    return WindLLResult(True,str(result.message),caps,dispatched,w-dispatched,
                        getattr(result,'mip_gap',None),getattr(result,'runtime_seconds',None),pred_abs)


def execute_wind_cap(*,prior_dispatched,wind_actual,cap,rated_mw=230.,ramp=40.):
    """Independent physical replay; actual wind revealed after optimized cap."""
    if not (-1e-7<=wind_actual<=rated_mw+1e-7):raise ValueError('wind actual outside rating')
    lo=max(0.,prior_dispatched-ramp);hi=min(rated_mw,prior_dispatched+ramp)
    if not lo-1e-5<=cap<=hi+1e-5:
        raise ValueError(f'cap slew violation: cap {cap}, prior dispatch {prior_dispatched}; allowed {lo}..{hi}')
    delivered=min(float(wind_actual),float(cap))
    if delivered>prior_dispatched+ramp+1e-5:raise AssertionError('positive wind ramp exceeded')
    downward_exception=max(0.,prior_dispatched-delivered-ramp)
    if downward_exception>1e-5 and wind_actual>=prior_dispatched-ramp-1e-5:
        raise AssertionError('Avoidable downward ramp exceeding 40MW')
    return delivered,wind_actual-delivered,downward_exception
