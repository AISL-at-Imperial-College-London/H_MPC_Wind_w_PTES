"""4-state LL MPC with 15min DA delivery tracking and ramped wind dispatch.

Gurobi (optional) solves the exact original convex MIQP. With only SciPy/HiGHS,
use a bounded-error tangential MILP approximation of precisely the same
quadratic tracking objective, rather than an unrelated absolute-value penalty.
The number of 5-minute steps can shrink at the end of a finite study episode.
"""
from dataclasses import dataclass
import numpy as np
from .physics import PlantParams,PlantState,MODES,EDGES,pwa_thermal_error_bounds
from .linear_mip import MIP,add_quadratic_epigraph
from .pwa import add_mapping
from .warm_start import shift_ll_binary_start,capture_ll_binary_seed

@dataclass
class LLSolution:
    success: bool
    status: int
    message: str
    objective_usd: float|None
    mip_gap: float|None
    charge_mw: np.ndarray|None
    discharge_mw: np.ndarray|None
    poi_forecast_mw: np.ndarray|None
    modes: list|None
    tank_mwhth: np.ndarray|None
    full_result: object=None
    solver_backend: str='unknown'
    max_quadratic_approx_error_usd: float=0.
    curtail_command_mw: np.ndarray|None=None  # legacy attribute; superseded by wind_cap_mw
    curtail_forecast_mw: np.ndarray|None=None
    predicted_tracking_abs_mwh: float=0.
    predicted_weighted_tracking_mwh: float=0.
    tracking_policy: str="none"
    first_stage_tracking_mwh: float|None=None
    wind_cap_mw: np.ndarray|None=None
    dispatched_wind_forecast_mw: np.ndarray|None=None
    certified_to_requested_gap: bool=True
    accepted_time_limit_incumbent: bool=False
    warm_start_seed: dict|None=None
    warm_start_submitted: bool=False
    warm_start_variables: int=0
    warm_start_skip_reason: str='disabled'

def solve_ll(wind_forecast_mw,rt_forecast_usd_mwh,state:PlantState,
             target_terminal_mwhth:float,prm:PlantParams=PlantParams(),
             terminal_beta_usd_per_mwhth2:float=1.,
             time_limit_s=30.,mip_rel_gap=.02,solver_backend='auto',
             quadratic_tangents=401,wind_uncertainty_margin_mw=0.,
             da_commitments_5min_mw=None,current_quarter_delivered_mwh=0.,
             quarter_offset_5min=0,tracking_policy="none",
             tracking_weight_usd_per_mwh=300.,tracking_budget_mwh=.25,
             curtailment_enabled=False,maximum_time_limit_gap=None,
             previous_solution=None,warm_start_enabled=False):
    """5-minute PTES LL; 'weighted' is a SINGLE-SHOT convex MIQP on Gurobi.

    A wind dispatch cap (when enabled) is ramp-limited relative to previous
    REALIZED dispatched wind, and predicted delivered wind=min(cap,availability).
    '-40 MW' is waived only when availability physically collapses.
    'none' with curtailment_enabled=False exactly preserves legacy LL physics.
    """
    w=np.asarray(wind_forecast_mw,float).ravel(); lam=np.asarray(rt_forecast_usd_mwh,float).ravel()
    N=len(w)
    if not 1<=N<=48 or len(lam)!=N:raise ValueError('LL expects 1..48 steps x 5 minutes')
    if not np.isfinite(w).all() or not np.isfinite(lam).all() or (w<0).any() or (w>prm.rated_wind_mw).any():
        raise ValueError('invalid forecasts')
    if not 0<=state.energy_mwhth<=prm.capacity_mwhth or not 0<=target_terminal_mwhth<=prm.capacity_mwhth:
        raise ValueError('SoC outside tank bounds')
    if terminal_beta_usd_per_mwhth2<0:raise ValueError('negative terminal penalty')
    if tracking_policy not in ('none','weighted','lexicographic'):
        raise ValueError('invalid tracking policy')
    if tracking_weight_usd_per_mwh<0 or tracking_budget_mwh<0:
        raise ValueError('negative tracking parameter')
    if maximum_time_limit_gap is not None and not (mip_rel_gap<=maximum_time_limit_gap<1.):
        raise ValueError('maximum_time_limit_gap must be >= requested gap and <1')
    if quarter_offset_5min not in (0,1,2):
        raise ValueError('quarter_offset_5min must be 0,1,2')
    if not np.isfinite(current_quarter_delivered_mwh):
        raise ValueError('invalid executed quarter energy')
    if da_commitments_5min_mw is None:
        q_da=np.full(N,np.nan)
    else:
        q_da=np.asarray(da_commitments_5min_mw,float).ravel()
        if len(q_da)!=N or not np.isfinite(q_da[~np.isnan(q_da)]).all():
            raise ValueError('invalid accepted DA vector')
    if tracking_policy!='none' and not np.isfinite(q_da).any():
        raise ValueError('DA tracking requested without accepted DA commitments')
    m=MIP();dt=1/12;retain=prm.daily_retention**(dt/24)
    # Rigorous true-polynomial-minus-PWA bounds, used only for safety envelopes.
    c_err_lo,c_err_hi=pwa_thermal_error_bounds("charge",prm)
    d_err_lo,d_err_hi=pwa_thermal_error_bounds("discharge",prm)
    y={};z={};pc={};pd={};qc={};qd={};E={};Elo={};Ehi={};curt={};ceff={};wdel={}
    curtail_disabled_variable=m.v('zero_curtailment',0.,0.) if not curtailment_enabled else None
    for t in range(N):
        for mode in MODES:
            y[t,mode]=m.v(f'y_{t}_{mode}',0,1,True,
                 obj=(dt*lam[t]*prm.standby_aux_mw if mode=='standby' else 0.))
        m.c({y[t,mode]:1 for mode in MODES},lo=1,hi=1)
        for a,b in EDGES:z[t,a,b]=m.v(f'z_{t}_{a}_{b}',0,1,True)
        m.c({z[t,a,b]:1 for a,b in EDGES},hi=1)
        for mode in MODES:
            co={y[t,mode]:1}
            if t>0:co[y[t-1,mode]]=-1;rhs=0.
            else:rhs=float(state.mode==mode)
            for a,b in EDGES:
                if b==mode:co[z[t,a,b]]=co.get(z[t,a,b],0)-1
                if a==mode:co[z[t,a,b]]=co.get(z[t,a,b],0)+1
            m.c(co,lo=rhs,hi=rhs)
        # At least six consecutive completed five-minute OFF intervals before restart.
        restart=z[t,'off','standby']
        if t<prm.off_min_steps and (state.mode!='off' or state.off_elapsed_steps<prm.off_min_steps-t):
            m.c({restart:1},hi=0)
        for prev in range(max(0,t-prm.off_min_steps),t):
            m.c({restart:1,y[prev,'off']:-1},hi=0)
        pc[t]=m.v(f'pc_{t}',0,prm.charge_max_mw,obj=dt*lam[t])
        pd[t]=m.v(f'pd_{t}',0,prm.discharge_max_mw,obj=-dt*lam[t])
        qc[t]=m.v(f'qc_{t}',0,prm.charge_max_mw*3)
        qd[t]=m.v(f'qd_{t}',0,prm.discharge_max_mw*4)
        E[t]=m.v(f'energy_after_{t}',0,prm.capacity_mwhth)
        Elo[t]=m.v(f'safe_energy_lower_{t}',0,prm.capacity_mwhth)
        Ehi[t]=m.v(f'safe_energy_upper_{t}',0,prm.capacity_mwhth)
        m.c({pc[t]:1,y[t,'charge']:-prm.charge_min_mw},lo=0)
        m.c({pc[t]:1,y[t,'charge']:-prm.charge_max_mw,
             z[t,'standby','charge']:(1-prm.startup_max_fraction)*prm.charge_max_mw},hi=0)
        m.c({pd[t]:1,y[t,'discharge']:-prm.discharge_min_mw},lo=0)
        m.c({pd[t]:1,y[t,'discharge']:-prm.discharge_max_mw,
             z[t,'standby','discharge']:(1-prm.startup_max_fraction)*prm.discharge_max_mw},hi=0)
        add_mapping(m,pc[t],qc[t],y[t,'charge'],'charge',prm,f'k{t}')
        add_mapping(m,pd[t],qd[t],y[t,'discharge'],'discharge',prm,f'k{t}')
        # Dispatchable wind: historical/predicted power is an availability ceiling.
        # For the controlled case cap[t] is the farm output reference.
        # Exactly represent dispatched_wind = min(w_forecast, cap[t]) in a MILP.
        # Each cap can change +/-40 MW from *previous dispatched wind*.
        # Unlike a cap-to-cap ramp, this prevents overshoot on wind recovery
        # after a lull and allows physical downward availability collapses.
        if curtailment_enabled:
            if t==0 and state.last_dispatched_wind_mw is None:
                raise ValueError('previous dispatched wind required for wind-ramp MPC')
            curt[t]=m.v(f'wind_dispatch_cap_{t}',0.,prm.rated_wind_mw)
            wdel[t]=m.v(f'wind_dispatch_forecast_{t}',0.,float(w[t]))
            ceff[t]=m.v(f'wind_curtail_forecast_{t}',0.,float(w[t]),obj=dt*lam[t])
            b=m.v(f'wind_above_dispatch_cap_{t}',0,1,True)
            big=prm.rated_wind_mw
            m.c({wdel[t]:1,ceff[t]:1},lo=float(w[t]),hi=float(w[t]))
            m.c({wdel[t]:1,curt[t]:-1},hi=0.)
            m.c({wdel[t]:1,curt[t]:-1,b:big},lo=0.)
            m.c({wdel[t]:1,b:-big},lo=float(w[t])-big)
            # Output-cap slew from the previously DELIVERED output, not
            # the previous cap: weather-induced shortfall is never fabricated.
            if t:
                m.c({curt[t]:1,wdel[t-1]:-1},
                    lo=-prm.wind_dispatch_ramp_mw_per_5min,
                    hi=prm.wind_dispatch_ramp_mw_per_5min)
            else:
                prior=float(state.last_dispatched_wind_mw)
                m.c({curt[t]:1},
                    lo=max(0.,prior-prm.wind_dispatch_ramp_mw_per_5min),
                    hi=min(prm.rated_wind_mw,prior+prm.wind_dispatch_ramp_mw_per_5min))
        else:
            curt[t]=None
            ceff[t]=curtail_disabled_variable
        m.c({pc[t]:-1,pd[t]:1,y[t,'standby']:-prm.standby_aux_mw,
             ceff[t]:-1},lo=-prm.poi_limit_mw-w[t],hi=prm.poi_limit_mw-w[t])
        if wind_uncertainty_margin_mw>0:
            m.c({pd[t]:1},hi=max(0.,prm.poi_limit_mw-w[t]-wind_uncertainty_margin_mw))
        co={E[t]:1,qc[t]:-dt,qd[t]:dt}
        if t>0:co[E[t-1]]=-retain;rhs=0
        else:rhs=retain*state.energy_mwhth
        m.c(co,lo=rhs,hi=rhs)
        # Conservatively bound the true nonlinear plant thermal energy, not just
        # the PWA surrogate. Without these, feasible PWA solutions can breach
        # tank limits by ~0.075 MWhth per step on discharge near empty and
        # ~0.023 MWhth per step on charge near full.
        lower={Elo[t]:1,qc[t]:-dt,qd[t]:dt,
               y[t,'charge']:-dt*c_err_lo,y[t,'discharge']:dt*d_err_hi}
        upper={Ehi[t]:1,qc[t]:-dt,qd[t]:dt,
               y[t,'charge']:-dt*c_err_hi,y[t,'discharge']:dt*d_err_lo}
        if t>0:
            lower[Elo[t-1]]=-retain;upper[Ehi[t-1]]=-retain
        m.c(lower,lo=rhs,hi=rhs)
        m.c(upper,lo=rhs,hi=rhs)
    # 15-minute energy tracking. Only COMPLETE forecast settlement quarters
    # enter the objective. The current partial quarter includes immutable
    # already-executed POI MWh. Skip an incomplete trailing horizon quarter.
    tracker=[]
    if tracking_policy!='none':
        buckets={}
        for j in range(N):buckets.setdefault((quarter_offset_5min+j)//3,[]).append(j)
        for k,idx in buckets.items():
            needed=3-quarter_offset_5min if k==0 else 3
            if len(idx)!=needed:continue
            commitments=q_da[idx]
            if np.isnan(commitments).any():continue  # beyond cleared DA hours
            if max(commitments)-min(commitments)>1e-6:
                raise ValueError('DA commitment changes within settlement quarter')
            scheduled=float(commitments[0])/4.
            carry=float(current_quarter_delivered_mwh) if k==0 else 0.
            weight=4. if k==0 else (2. if k<=3 else 1.)
            plus=m.v(f'da_long_mwh_quarter_{k}',0,np.inf)
            minus=m.v(f'da_short_mwh_quarter_{k}',0,np.inf)
            co={plus:1.,minus:-1.}
            for j in idx:
                for v,coef in ((pc[j],dt),(pd[j],-dt),(ceff[j],dt),
                               (y[j,'standby'],dt*prm.standby_aux_mw)):
                    co[v]=co.get(v,0.)+coef
            rhs=carry+dt*sum(w[j] for j in idx)-scheduled
            m.c(co,lo=rhs,hi=rhs)
            tracker.extend(((plus,weight),(minus,weight)))
    quadratic_terms=[(E[N-1],target_terminal_mwhth,terminal_beta_usd_per_mwhth2)]
    if solver_backend not in ('auto','gurobi','highs'):
        raise ValueError('solver_backend must be auto, gurobi or highs')
    result=None;used='';primary_result=None;primary_val=None
    if tracking_policy=='lexicographic' and tracker:
        # Primary problem has units of WEIGHTED MWh, not USD.
        economics=list(m.obj)
        m.obj=[0.]*len(m.obj)
        for v,weight in tracker:m.obj[v]=weight
        if solver_backend in ('auto','gurobi'):
            try:
                primary_result=m.solve_gurobi(time_limit_s,mip_rel_gap)
            except Exception:
                if solver_backend=='gurobi':raise
        if primary_result is None:primary_result=m.solve(time_limit_s,mip_rel_gap)
        if not primary_result.success or primary_result.x is None:
            # No controller action may be executed from an uncertified stage 1.
            from types import SimpleNamespace
            r=primary_result
            return LLSolution(False,r.status,'Tracking stage 1: '+str(r.message),
                None,getattr(r,'mip_gap',None),None,None,None,None,None,r,
                'tracking_stage_1',0.,tracking_policy=tracking_policy)
        primary_val=float(sum(weight*primary_result.x[v] for v,weight in tracker))
        m.c({v:weight for v,weight in tracker},hi=primary_val+tracking_budget_mwh)
        m.obj=economics
    elif tracking_policy=='weighted':
        for v,weight in tracker:m.obj[v]+=tracking_weight_usd_per_mwh*weight

    # Rolling Gurobi start includes *only* shifted discrete mode/segment
    # choices; the revised wind, power and thermal trajectories remain free.
    mip_start={}
    start_reason='disabled'
    if warm_start_enabled and solver_backend in ('auto','gurobi'):
        mip_start,start_reason=shift_ll_binary_start(
            previous_solution,state,m.names,m.integer,prm.off_min_steps)
    elif warm_start_enabled:
        start_reason='non_gurobi_backend'
    if solver_backend in ('auto','gurobi'):
        try:
            result=m.solve_exact_miqp(quadratic_terms,time_limit_s,mip_rel_gap,
                                      maximum_time_limit_gap=maximum_time_limit_gap,
                                      **({'mip_start':mip_start} if mip_start else {}))
            used='gurobi_exact_miqp'
        except Exception:
            if solver_backend=='gurobi':raise
            result=None
    max_error=0.
    if result is None:
        max_error=add_quadratic_epigraph(m,E[N-1],target_terminal_mwhth,
              terminal_beta_usd_per_mwhth2,prm.capacity_mwhth,n_tangents=quadratic_tangents)
        result=m.solve(time_limit_s,mip_rel_gap)
        used='highs_quadratic_epigraph_milp'

    if result.x is None:
        return LLSolution(False,result.status,str(result.message),None,None,None,None,
                          None,None,None,result,used,max_error)
    xx=result.x
    pcs=np.array([xx[pc[t]] for t in range(N)])
    pds=np.array([xx[pd[t]] for t in range(N)])
    modes=[max(MODES,key=lambda mode:xx[y[t,mode]]) for t in range(N)]
    cap=(np.array([xx[curt[t]] for t in range(N)]) if curtailment_enabled else None)
    c_eff=(np.maximum(0.,w-np.minimum(w,cap)) if curtailment_enabled else np.zeros(N))
    delivered=w-c_eff
    poi=delivered-pcs+pds-prm.standby_aux_mw*np.array([mode=='standby' for mode in modes])
    ee=np.array([xx[E[t]] for t in range(N)])
    true_penalty=float(terminal_beta_usd_per_mwhth2*(ee[-1]-target_terminal_mwhth)**2)
    # Report the true quadratic value (not the PWL approximation).
    true_cost=float(np.sum(dt*lam*(pcs-pds+c_eff+
                  prm.standby_aux_mw*np.array([a=='standby' for a in modes])))+true_penalty)
    raw_tracking=float(sum(xx[v] for v,_ in tracker))
    weighted_tracking=float(sum(weight*xx[v] for v,weight in tracker))
    if tracking_policy=='weighted':true_cost+=tracking_weight_usd_per_mwh*weighted_tracking
    return LLSolution(bool(result.success),result.status,str(result.message),true_cost,
                      getattr(result,'mip_gap',None),pcs,pds,poi,modes,ee,result,used,max_error,
                      None,c_eff,raw_tracking,weighted_tracking,tracking_policy,primary_val,
                      cap,delivered,
                      getattr(result,'certified_to_requested_gap',bool(result.success)),
                      getattr(result,'accepted_time_limit_incumbent',False),
                      capture_ll_binary_seed(m.names,m.integer,xx) if result.success else {},
                      bool(mip_start and used=='gurobi_exact_miqp'),
                      int(getattr(result,'mip_start_variables',len(mip_start) if used=='gurobi_exact_miqp' else 0)),
                      start_reason)
