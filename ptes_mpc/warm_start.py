"""Safe *partial* Gurobi MIP starts from a previous rolling-horizon LL plan.

Only mode/transition/PWA segment binaries are carried over. Continuous
quantities (wind, electric power, thermal inventories, tracking slacks) are
reoptimized because the measured initial conditions and forecasts change.
Gurobi is free to repair/discard an infeasible partial start, and no variable
is fixed by this module. The final appended interval is intentionally free.
"""
from __future__ import annotations
import math
from .physics import MODES, EDGES


def shift_ll_binary_start(previous_solution, state, names, is_integer, off_min_steps):
    """Return (index->binary initial value, reason) for *one* five-minute shift.

    No lookahead: previous_solution is a previously computed LL optimization,
    and state is the actual measured state *after* executing its first action.
    In case of a regulatory mode override or invalid transition, skip the start.
    """
    if previous_solution is None:
        return {}, 'first_solve'
    if not bool(getattr(previous_solution, 'success', False)):
        return {}, 'previous_solve_failed'
    modes = getattr(previous_solution, 'modes', None)
    values = getattr(previous_solution, 'warm_start_seed', None)
    if not modes or not isinstance(values, dict) or not values:
        return {}, 'previous_plan_missing'
    if modes[0] != state.mode:
        return {}, 'executed_mode_differs_from_previous_plan'
    n = sum(1 for v in names if v.startswith('y_') and v.endswith('_off'))
    length = min(n, len(modes)-1)
    if length <= 0:
        return {}, 'no_overlapping_forecast_steps'
    # Require the proposed shift to respect executed mode and the six-step off timer.
    oldmode = state.mode
    off_elapsed = int(state.off_elapsed_steps)
    for t in range(length):
        mode = modes[t+1]
        if mode != oldmode and (oldmode,mode) not in EDGES:
            return {}, 'shifted_mode_transition_invalid'
        if oldmode == 'off' and mode == 'standby' and off_elapsed < off_min_steps:
            return {}, 'shifted_minimum_off_time_invalid'
        off_elapsed = off_elapsed+1 if mode == 'off' else 0
        oldmode = mode
    by_name = {name:i for i,name in enumerate(names) if is_integer[i]}
    start = {}
    for t in range(length):
        mode = modes[t+1]
        prior = state.mode if t==0 else modes[t]
        for candidate in MODES:
            name = f'y_{t}_{candidate}'
            if name in by_name:
                start[by_name[name]] = float(candidate==mode)
        for a,b in EDGES:
            name = f'z_{t}_{a}_{b}'
            if name in by_name:
                start[by_name[name]] = float(prior==a and mode==b)
        for family in ('charge','discharge'):
            prefix=f'k{t}_{family}_seg'
            oldprefix=f'k{t+1}_{family}_seg'
            new_segments=sorted((name for name in by_name if name.startswith(prefix)),
                                key=lambda s:int(s[len(prefix):]))
            if not new_segments:
                continue
            if mode != family:
                for name in new_segments:
                    start[by_name[name]]=0.
                continue
            # Segment choices may be at an end-point; reuse the previous
            # segment, but leave all continuous segment extents free.
            prev_segments = [values.get(oldprefix+name[len(prefix):]) for name in new_segments]
            if any(v is None or not math.isfinite(float(v)) for v in prev_segments):
                continue
            chosen=max(range(len(prev_segments)),key=lambda i:float(prev_segments[i]))
            if float(prev_segments[chosen])<0.5:
                continue
            for i,name in enumerate(new_segments):
                start[by_name[name]]=float(i==chosen)
    return start, ('shifted_partial_binary_start' if start else 'no_matching_binaries')


def capture_ll_binary_seed(names, is_integer, solution_vector):
    """Persist named mode/PWA binaries, never time-dependent continuous states."""
    if solution_vector is None:
        return {}
    prefixes = ('y_', 'z_', 'k')
    result = {}
    for i, name in enumerate(names):
        if is_integer[i] and name.startswith(prefixes) and (
            name.startswith(('y_', 'z_')) or '_seg' in name):
            val=float(solution_vector[i])
            if math.isfinite(val) and abs(val-round(val))<1e-4:
                result[name]=float(round(val))
    return result
