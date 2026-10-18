"""Quasi-steady PTES energy/state-machine replay with commanded/actual separation.

It is NOT a thermodynamic transient simulator. The regulatory layer enforces
instantaneous export by backing down commanded PTES discharge. If this requires
sub-minimum discharge it forces the *actual* machine to OFF, resets the minimum
six-step off-time clock, and updates the thermal tank using actual power.
"""
from __future__ import annotations
from dataclasses import dataclass

MODES = ('off', 'standby', 'charge', 'discharge')
EDGES = (('off', 'standby'), ('standby', 'off'), ('standby', 'charge'),
         ('standby', 'discharge'), ('charge', 'standby'), ('discharge', 'standby'),
         ('charge', 'off'), ('discharge', 'off'))

@dataclass(frozen=True)
class PlantParams:
    rated_wind_mw: float = 230.0
    poi_limit_mw: float = 230.0
    gamma: float = 230/300
    charge_min_mw: float = 57.4 * 230/300
    charge_max_mw: float = 100 * 230/300
    discharge_min_mw: float = 51 * 230/300
    discharge_max_mw: float = 220 * 230/300
    capacity_mwhth: float = 3503 * 230/300
    standby_aux_mw: float = 230/300
    daily_retention: float = .99
    off_min_steps: int = 6
    startup_max_fraction: float = .7
    pwa_segments: int = 5
    wind_dispatch_ramp_mw_per_5min: float = 40.0

    def __post_init__(self):
        assert 0 < self.daily_retention <= 1 and 0 < self.gamma <= 1
        assert self.charge_min_mw < self.startup_max_fraction * self.charge_max_mw
        assert self.discharge_min_mw < self.startup_max_fraction * self.discharge_max_mw
        assert 0 < self.wind_dispatch_ramp_mw_per_5min <= self.rated_wind_mw

@dataclass(frozen=True)
class PlantState:
    energy_mwhth: float
    mode: str = 'off'
    off_elapsed_steps: int = 6
    last_dispatched_wind_mw: float|None = None

    def __post_init__(self):
        assert self.mode in MODES and self.energy_mwhth >= 0 and self.off_elapsed_steps >= 0
        assert self.last_dispatched_wind_mw is None or self.last_dispatched_wind_mw >= 0

def thermal_charge_mw(p_mwe: float, prm: PlantParams) -> float:
    if p_mwe <= 0: return 0.
    x = p_mwe / prm.gamma
    return prm.gamma * (.0002*x**3 - .057*x**2 + 7.1783*x - 188.56)

def thermal_discharge_mw(p_mwe: float, prm: PlantParams) -> float:
    if p_mwe <= 0: return 0.
    x = p_mwe / prm.gamma
    return prm.gamma * (-.0041*x*x + 3.2568*x + 35.212)

def pwa_breakpoints(mode: str, prm: PlantParams):
    import numpy as np
    lo,hi,fun = ((prm.charge_min_mw,prm.charge_max_mw,thermal_charge_mw) if mode=='charge'
                 else (prm.discharge_min_mw,prm.discharge_max_mw,thermal_discharge_mw))
    xs = np.linspace(lo,hi,prm.pwa_segments+1)
    ys = np.array([fun(x,prm) for x in xs])
    return xs,ys

def pwa_thermal_error_bounds(mode: str, prm: PlantParams):
    """Exact extrema of true polynomial minus 5-segment linear interpolation.

    Evaluate endpoints and real roots of (polynomial derivative - chord slope)
    on each PWA segment; a small outward tolerance guards roundoff.
    The returned bounds are in MW thermal, not MWh.
    """
    import numpy as np
    xs,ys=pwa_breakpoints(mode,prm)
    fun=thermal_charge_mw if mode=='charge' else thermal_discharge_mw
    values=[]
    for i in range(len(xs)-1):
        left,right=float(xs[i]),float(xs[i+1]);slope=float((ys[i+1]-ys[i])/(right-left))
        if mode=='charge':
            roots=np.roots([.0006/prm.gamma**2,-.114/prm.gamma,7.1783-slope])
        elif mode=='discharge':
            roots=np.roots([-.0082/prm.gamma,3.2568-slope])
        else:
            raise ValueError('unknown PWA mode')
        candidates=[left,right]+[float(r.real) for r in roots if abs(r.imag)<1e-10 and left<=r.real<=right]
        values += [fun(x,prm)-(ys[i]+slope*(x-left)) for x in candidates]
    return min(values)-1e-9,max(values)+1e-9


def execute_step(state:PlantState, mode:str, p_charge:float, p_discharge:float,
                 wind_real:float, prm:PlantParams, dt_hr=1/12,
                 allow_safety_override=True, wind_dispatch_cap_mw=None):
    """Execute a *pre-issued* farm dispatch cap against historical AVAILABLE wind.

    When enabled, the cap must stay within +/-40 MW of the previous **realized
    dispatched** wind; actual delivered wind is min(availability,cap).
    An availability collapse is allowed to violate the -40 MW output ramp:
    never fabricate power. The independent PTES regulator then checks POI.
    When cap=None, return the original uncurtailed historical-wind baseline.
    """
    if mode not in MODES: raise ValueError('unknown mode')
    if mode != state.mode and (state.mode,mode) not in EDGES:
        raise ValueError('forbidden transition')
    if state.mode=='off' and mode!='off' and state.off_elapsed_steps<prm.off_min_steps:
        raise ValueError('minimum off time')
    if not 0 <= wind_real <= prm.rated_wind_mw:
        raise ValueError('wind outside capacity')
    previous=state.last_dispatched_wind_mw
    if wind_dispatch_cap_mw is None:
        cap=float(prm.rated_wind_mw)
        dispatched=float(wind_real)
    else:
        cap=float(wind_dispatch_cap_mw)
        if not (-1e-6<=cap<=prm.rated_wind_mw+1e-6):
            raise ValueError('wind output cap outside nameplate bounds')
        if previous is None:
            raise ValueError('missing preceding dispatched wind for ramp-controlled dispatch')
        hi=min(prm.rated_wind_mw,previous+prm.wind_dispatch_ramp_mw_per_5min)
        lo=max(0.,previous-prm.wind_dispatch_ramp_mw_per_5min)
        if not lo-1e-5<=cap<=hi+1e-5:
            raise ValueError(f'wind output cap ramp violation: previous={previous}, cap={cap}, bounds={lo},{hi}')
        dispatched=min(float(wind_real),cap)
        if dispatched>previous+prm.wind_dispatch_ramp_mw_per_5min+1e-5:
            raise AssertionError('realized positive wind dispatch ramp exceeded bound')
    curtailed=float(wind_real)-dispatched
    wind_delivered=dispatched
    change=(None if previous is None else float(dispatched-previous))
    downward_exception=max(0.,-(change or 0.)-prm.wind_dispatch_ramp_mw_per_5min)
    if downward_exception>1e-5 and wind_real>=previous-prm.wind_dispatch_ramp_mw_per_5min-1e-5:
        raise AssertionError('downward wind dispatch violation without availability shortfall')
    # An exact MIQP incumbent can have a *tiny* nonzero continuous power
    # even when its associated binary mode rounds to zero. Gurobi's default
    # integrality tolerance is 1e-5, so pc <= charge_max*y_charge can allow
    # an inactive residual of ~charge_max*1e-5 MW, plus LP row tolerance.
    # The old 1e-8 MW check can spuriously reject a valid incumbent; this
    # is a likely explanation of the winter half-capacity replay error.
    # Accept ONLY that bounded
    # numerical residue; anything larger remains a HARD physical failure.
    # Do not alter the MIQP, tank model, mode selection or economic objective.
    inactive_charge_tol_mw=prm.charge_max_mw*1.01e-5+2e-6
    inactive_discharge_tol_mw=prm.discharge_max_mw*1.01e-5+2e-6
    if mode!='charge' and abs(p_charge)>inactive_charge_tol_mw:
        raise ValueError(
            f'charge outside charge mode: mode={mode}, charge={p_charge:.12g} MW, '
            f'roundoff_limit={inactive_charge_tol_mw:.12g} MW, '
            f'discharge={p_discharge:.12g} MW, previous_mode={state.mode}')
    if mode!='discharge' and abs(p_discharge)>inactive_discharge_tol_mw:
        raise ValueError(
            f'discharge outside discharge mode: mode={mode}, '
            f'discharge={p_discharge:.12g} MW, '
            f'roundoff_limit={inactive_discharge_tol_mw:.12g} MW, '
            f'charge={p_charge:.12g} MW, previous_mode={state.mode}')
    # Every tolerated inactive-mode residue is explicitly projected to zero.
    # The existing numerical_projection_mw log records the adjustment.
    # Otherwise even tiny nonzero power can activate a polynomial intercept
    # and create physically spurious tank flows.
    raw_charge,raw_discharge=float(p_charge),float(p_discharge)
    if mode!='charge':p_charge=0.0
    if mode!='discharge':p_discharge=0.0
    # Project only HiGHS numerical feasibility tolerances (<1e-3 MW) to exact
    # physical part-load boundaries. Never project a materially infeasible command.
    if mode == 'charge' and prm.charge_min_mw-1e-3 <= p_charge < prm.charge_min_mw:
        p_charge=prm.charge_min_mw
    if mode == 'discharge' and prm.discharge_min_mw-1e-3 <= p_discharge < prm.discharge_min_mw:
        p_discharge=prm.discharge_min_mw
    if mode == 'charge' and not (prm.charge_min_mw-1e-7<=p_charge<=prm.charge_max_mw+1e-7):
        raise ValueError(f'charging part-load {p_charge} in mode {mode}, limits [{prm.charge_min_mw},{prm.charge_max_mw}]')
    if mode == 'discharge' and not (prm.discharge_min_mw-1e-7<=p_discharge<=prm.discharge_max_mw+1e-7):
        raise ValueError(f'discharging part-load {p_discharge} limits [{prm.discharge_min_mw},{prm.discharge_max_mw}]')
    if state.mode=='standby' and mode in ('charge','discharge'):
        mx=prm.charge_max_mw if mode=='charge' else prm.discharge_max_mw
        if (p_charge if mode=='charge' else p_discharge)>prm.startup_max_fraction*mx+1e-7:
            raise ValueError('startup 70% derate')

    commanded_mode=mode
    commanded_charge=float(p_charge)
    commanded_discharge=float(p_discharge)
    aux=prm.standby_aux_mw if mode=='standby' else 0.
    requested_poi=wind_delivered-p_charge+p_discharge-aux
    intervention='none'
    if requested_poi>prm.poi_limit_mw+1e-8 and allow_safety_override and mode=='discharge':
        allowed=max(0.,prm.poi_limit_mw-wind_delivered+aux)
        if allowed>=prm.discharge_min_mw-1e-7:
            p_discharge=min(float(p_discharge),allowed)
            intervention='partial_discharge_backdown'
        else:
            # An actively generating machine is shut off, and the off clock is reset.
            # Charge/discharge->off and standby->off are allowed thesis transitions.
            mode='off'
            p_discharge=0.
            aux=0.
            intervention='subminimum_forced_off'
    poi=wind_delivered-p_charge+p_discharge-aux
    if abs(poi)>prm.poi_limit_mw+1e-7:
        raise ValueError('realized POI violation; regulatory discharge clipping insufficient')
    retain=prm.daily_retention**(dt_hr/24)
    energy=(state.energy_mwhth*retain
            +dt_hr*(thermal_charge_mw(p_charge,prm)-thermal_discharge_mw(p_discharge,prm)))
    if not (-1e-7<=energy<=prm.capacity_mwhth+1e-7):
        raise ValueError('realized tank energy violation')
    if mode=='off':
        elapsed=(min(prm.off_min_steps,state.off_elapsed_steps+1)
                 if state.mode=='off' else 1)
    else:
        elapsed=0
    next_state=PlantState(max(0.,min(energy,prm.capacity_mwhth)),mode,elapsed,dispatched)
    return next_state, {
        'commanded_mode':commanded_mode, 'actual_mode':mode,
        'numerical_projection_mw':abs(raw_charge-p_charge)+abs(raw_discharge-p_discharge),
        'commanded_charge_mw':commanded_charge,
        'commanded_discharge_mw':commanded_discharge,
        'requested_poi_mw':requested_poi,
        'available_wind_mw':float(wind_real),
        'wind_dispatch_cap_mw':cap,'wind_dispatch_change_mw':change,
        'wind_downward_ramp_exception_mw':downward_exception,
        'curtailed_wind_mw':curtailed,'dispatched_wind_mw':wind_delivered,
        'poi_mw':poi,'charge_mw':float(p_charge),
        'discharge_mw':float(p_discharge),
        'discharge_clip_mw':commanded_discharge-float(p_discharge),
        'standby_aux_mw':aux,'safety_override':intervention!='none',
        'regulatory_action':intervention,
        'energy_mwh':poi*dt_hr,'tank_mwhth':next_state.energy_mwhth,
    }
