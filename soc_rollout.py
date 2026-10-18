"""Causal bid-to-delivery SoC prediction by forward rollout (point estimate).

Replaces the "previous plan reference change" bridge in soc_handover.py.

At the HL bid time t (D 09:30 local) the delivery-midnight inventory depends on
  * the measured state at t (energy, mode, off-timer),
  * the ALREADY ACCEPTED DA commitments for the rest of day D (fixed, known),
  * wind between t and midnight, and how the LL responds to it.
This module rolls the plant forward over that gap with
  * a POINT wind forecast for the gap (LL 5-min XGBoost for the first 4 h,
    then exponential relaxation towards an anchor level), and
  * a model of the LL policy (default: a solver-free DA-tracking rule; or any
    callable, e.g. a shadow call of the new DA-tracking LL MPC),
using exactly the same plant model as the closed loop (physics.execute_step),
with forecast wind standing in for realised wind.

Causality: only measurements strictly before t, commitments already accepted,
and DA prices already cleared are used. `oracle_rollout` (realised gap wind)
is a POST-HOC diagnostic only and must never feed a controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ptes_mpc.physics import (PlantParams, PlantState, execute_step,
                              thermal_charge_mw, thermal_discharge_mw)

STEP = pd.Timedelta(minutes=5)
DT_H = 1 / 12


# ----------------------------------------------------------------------------
# results
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GapRolloutResult:
    issue_utc: pd.Timestamp
    delivery_utc: pd.Timestamp
    policy: str
    wind_source: str
    initial_soc_mwhth: float
    predicted_soc_mwhth: float
    predicted_mode_at_delivery: str
    n_steps: int
    gap_wind_forecast_mean_mw: float
    anchor_wind_mw: Optional[float]
    gap_commitment_mean_mw: float
    expected_shortfall_mwh: float      # sum max(q - POI, 0)/12 over the gap
    expected_surplus_mwh: float        # sum max(POI - q, 0)/12 over the gap
    regulatory_interventions: int
    curtailed_wind_mwh: float
    wind_downward_ramp_exceptions: int
    trajectory: pd.DataFrame = field(repr=False)

    def log_fields(self, prefix='rollout_') -> dict:
        """Flat fields for hl_runs / CSV logging (no trajectory)."""
        return {f'{prefix}policy': self.policy, f'{prefix}wind_source': self.wind_source,
                f'{prefix}initial_soc_mwhth': self.initial_soc_mwhth,
                f'{prefix}predicted_soc_mwhth': self.predicted_soc_mwhth,
                f'{prefix}predicted_mode_at_delivery': self.predicted_mode_at_delivery,
                f'{prefix}steps': self.n_steps,
                f'{prefix}gap_wind_forecast_mean_mw': self.gap_wind_forecast_mean_mw,
                f'{prefix}anchor_wind_mw': self.anchor_wind_mw,
                f'{prefix}gap_commitment_mean_mw': self.gap_commitment_mean_mw,
                f'{prefix}expected_shortfall_mwh': self.expected_shortfall_mwh,
                f'{prefix}expected_surplus_mwh': self.expected_surplus_mwh,
                f'{prefix}regulatory_interventions': self.regulatory_interventions,
                f'{prefix}curtailed_wind_mwh': self.curtailed_wind_mwh,
                f'{prefix}wind_downward_ramp_exceptions': self.wind_downward_ramp_exceptions}


# ----------------------------------------------------------------------------
# gap wind point forecast
# ----------------------------------------------------------------------------
def gap_wind_forecast(issue_utc, n_steps, short_term_48, anchor_mw, *,
                      relax_hours=4.0, prm=PlantParams()):
    """5-min point forecast over the whole gap.

    Steps 0..47 use the LL short-term forecast (issued at t). Beyond 4 h the
    path relaxes exponentially from the last short-term value towards
    `anchor_mw` with time constant `relax_hours` (an untuned assumption; relaxation time must be finite and positive).
    """
    if n_steps < 1 or not np.isfinite(relax_hours) or relax_hours <= 0:
        raise ValueError('gap length and relaxation time must be positive and finite')
    s = np.asarray(short_term_48, float).ravel()
    if s.shape != (48,) or not np.isfinite(s).all():
        raise ValueError('short-term wind forecast must be 48 finite values')
    path = np.empty(n_steps)
    k = min(48, n_steps)
    path[:k] = s[:k]
    if n_steps > 48:
        if anchor_mw is None or not np.isfinite(anchor_mw):
            raise ValueError('anchor wind needed beyond the 4-h short-term horizon')
        tau_h = (np.arange(48, n_steps) - 47) / 12.0
        decay = np.exp(-tau_h / relax_hours) if np.isfinite(relax_hours) else np.ones_like(tau_h)
        path[48:] = anchor_mw + (s[-1] - anchor_mw) * decay
    return np.clip(path, 0.0, prm.rated_wind_mw)


# ----------------------------------------------------------------------------
# LL policy models
# ----------------------------------------------------------------------------
def _max_discharge_for_tank(energy, prm, upper):
    """Largest discharge power <= upper whose 5-min thermal draw fits the tank."""
    avail = energy * prm.daily_retention ** (DT_H / 24)
    if thermal_discharge_mw(prm.discharge_min_mw, prm) * DT_H > avail:
        return 0.0
    if thermal_discharge_mw(upper, prm) * DT_H <= avail:
        return upper
    lo, hi = prm.discharge_min_mw, upper
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if thermal_discharge_mw(mid, prm) * DT_H <= avail else (lo, mid)
    return lo


def _max_charge_for_tank(energy, prm, upper):
    room = prm.capacity_mwhth - energy * prm.daily_retention ** (DT_H / 24)
    if thermal_charge_mw(prm.charge_min_mw, prm) * DT_H > room:
        return 0.0
    if thermal_charge_mw(upper, prm) * DT_H <= room:
        return upper
    lo, hi = prm.charge_min_mw, upper
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if thermal_charge_mw(mid, prm) * DT_H <= room else (lo, mid)
    return lo


def tracking_rule_policy(t, state: PlantState, wind_path, q_path, price_path, prm=PlantParams()):
    """Solver-free model of a DA-tracking LL (one step).

    Net PTES export needed to meet the commitment: need = q - wind.
    * Discharge if need > discharge_min/2 (overshooting at min load is then a
      smaller absolute deviation than idling); charge symmetrically.
    * Power limited by rating, 70% start-up derate from standby, POI headroom
      and tank energy/room for this 5-min step.
    * Legal transitions only (no direct charge<->discharge; off->standby after
      the minimum off time). Idle from an active mode goes to standby.
    Ignores the LL's secondary RT-revenue and SoC-reference terms: it is a
    deliberate approximation of 'track DA first'.
    """
    w = float(wind_path[0]); q = float(q_path[0])
    need = q - w
    E = state.energy_mwhth
    want = 'idle'
    if need > 0.5 * prm.discharge_min_mw:
        want = 'discharge'
    elif -need > 0.5 * prm.charge_min_mw:
        want = 'charge'

    def discharge_power(from_standby):
        upper = prm.discharge_max_mw * (prm.startup_max_fraction if from_standby else 1.0)
        upper = min(upper, prm.poi_limit_mw - w)          # never request a POI breach
        if upper < prm.discharge_min_mw:
            return 0.0
        p = min(max(need, prm.discharge_min_mw), upper)
        p = min(p, _max_discharge_for_tank(E, prm, upper))
        return p if p >= prm.discharge_min_mw else 0.0

    def charge_power(from_standby):
        upper = prm.charge_max_mw * (prm.startup_max_fraction if from_standby else 1.0)
        p = min(max(-need, prm.charge_min_mw), upper)
        p = min(p, _max_charge_for_tank(E, prm, upper))
        return p if p >= prm.charge_min_mw else 0.0

    m = state.mode
    if m == 'off':
        if want != 'idle' and state.off_elapsed_steps >= prm.off_min_steps:
            return 'standby', 0.0, 0.0
        return 'off', 0.0, 0.0
    if m == 'standby':
        if want == 'discharge':
            p = discharge_power(True)
            if p > 0:
                return 'discharge', 0.0, p
        if want == 'charge':
            p = charge_power(True)
            if p > 0:
                return 'charge', p, 0.0
        return 'standby', 0.0, 0.0
    if m == 'discharge':
        if want == 'discharge':
            p = discharge_power(False)
            if p > 0:
                return 'discharge', 0.0, p
        return 'standby', 0.0, 0.0
    if m == 'charge':
        if want == 'charge':
            p = charge_power(False)
            if p > 0:
                return 'charge', p, 0.0
        return 'standby', 0.0, 0.0
    raise ValueError(f'unknown mode {m}')


def make_shadow_ll_policy(solve_fn: Callable, *, horizon=48, resolve_every=3):
    """Wrap the real (new) LL MPC as a rollout policy.

    solve_fn(state, wind48, price48, q48) must return an object with
    `.modes`, `.charge_mw`, `.discharge_mw` (as solve_ll does). The plan is
    re-solved every `resolve_every` steps and followed open-loop in between;
    prices are the cleared DA prices of each hour (the best causal RT
    expectation 4-14 h ahead). Expect roughly 58 solves per bid at the default.
    """
    cache = {'plan': None, 'k': 0}

    def reset():
        cache['plan'] = None; cache['k'] = 0

    def policy(t, state, wind_path, q_path, price_path, prm=PlantParams()):
        n = min(horizon, len(wind_path))
        pad = lambda a: np.r_[a[:n], np.repeat(a[n - 1], horizon - n)] if n < horizon else np.asarray(a[:horizon])
        if cache['plan'] is None or cache['k'] >= resolve_every:
            cache['plan'] = solve_fn(state, pad(wind_path), pad(price_path), pad(q_path))
            cache['k'] = 0
        sol = cache['plan']; j = cache['k']; cache['k'] += 1
        if sol is None or sol.modes is None:
            raise RuntimeError(f'shadow LL failed at {t}')
        return sol.modes[j], float(sol.charge_mw[j]), float(sol.discharge_mw[j])

    policy.__name__ = f'shadow_ll_every{resolve_every}'
    policy.reset = reset          # rollout_delivery_soc resets before every rollout
    return policy


# ----------------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------------
def _commitment_path(times, commitments_hourly):
    q = pd.Series(commitments_hourly, dtype=float)
    q.index = pd.DatetimeIndex(q.index).tz_convert('UTC')
    hours = times.floor('h')
    vals = q.reindex(hours).to_numpy(float)
    if not np.isfinite(vals).all():
        missing = hours[~np.isfinite(vals)][0]
        raise ValueError(f'no accepted DA commitment for gap hour {missing}')
    return vals


def _da_path(times, da_prices_hourly):
    d = pd.Series(da_prices_hourly, dtype=float)
    d.index = pd.DatetimeIndex(d.index).tz_convert('UTC')
    vals = d.reindex(times.floor('h')).to_numpy(float)
    if not np.isfinite(vals).all():
        raise ValueError('cleared DA price missing in the gap')
    return vals


def rollout_delivery_soc(*, issue_utc, delivery_utc, state: PlantState,
                         commitments_hourly, da_prices_hourly,
                         wind_path, wind_source: str, anchor_wind_mw=None,
                         policy='tracking_rule', prm=PlantParams(),
                         curtailment_enabled=False) -> GapRolloutResult:
    """Roll the plant from issue to delivery midnight under a policy model.

    wind_path : 5-min wind for every step in [issue, delivery) — a causal
                point forecast for control; realised wind only in oracle_rollout.
    """
    t0 = pd.Timestamp(issue_utc).tz_convert('UTC')
    t1 = pd.Timestamp(delivery_utc).tz_convert('UTC')
    if t0.second or t0.microsecond or t0.minute % 5 or not t0 < t1:
        raise ValueError('issue must be a 5-min boundary before delivery')
    times = pd.date_range(t0, t1 - STEP, freq='5min', tz='UTC')
    w = np.asarray(wind_path, float).ravel()
    if len(w) != len(times) or not np.isfinite(w).all():
        raise ValueError(f'wind path must have {len(times)} finite 5-min values')
    q = _commitment_path(times, commitments_hourly)
    lam = _da_path(times, da_prices_hourly)
    fn = tracking_rule_policy if policy == 'tracking_rule' else policy
    if not callable(fn):
        raise ValueError("policy must be 'tracking_rule' or a callable")
    if hasattr(fn, 'reset'):
        fn.reset()                # stateful policies (shadow LL) start clean
    name = policy if isinstance(policy, str) else getattr(policy, '__name__', 'custom')

    cur = state
    if curtailment_enabled and cur.last_dispatched_wind_mw is None:
        raise ValueError('wind ramp rollout needs preceding realised dispatched wind in PlantState')
    rows = []
    for j, t in enumerate(times):
        command=fn(t, cur, w[j:], q[j:], lam[j:], prm)
        if len(command)==3:
            mode, pc, pdis = command
            explicit_cap = None
        elif len(command)==4:
            mode, pc, pdis, explicit_cap = command
        else:
            raise ValueError('policy must return mode, charge, discharge[, wind output cap]')
        cap=None
        if curtailment_enabled:
            previous=cur.last_dispatched_wind_mw
            lo=max(0.,previous-prm.wind_dispatch_ramp_mw_per_5min)
            hi=min(prm.rated_wind_mw,previous+prm.wind_dispatch_ramp_mw_per_5min)
            if explicit_cap is not None:
                # A custom policy supplies its own cap; never silently repair an illegal command.
                cap=float(explicit_cap)
            elif policy=='tracking_rule':
                # Storage is allocated against uncurtailed wind first; curtail only
                # remaining surplus relative to the accepted DA position.
                aux=prm.standby_aux_mw if mode=='standby' else 0.
                cap=float(np.clip(q[j]+pc-pdis+aux,0.,prm.rated_wind_mw))
                cap=float(np.clip(cap,lo,hi))
            else:
                # Generic custom policy not specifying curtailment: preserve as much
                # generation as permitted by the dispatch-output ramp.
                cap=float(hi)
        nxt, log = execute_step(cur, mode, pc, pdis, float(w[j]), prm,
                                allow_safety_override=True,wind_dispatch_cap_mw=cap)
        rows.append(dict(time_utc=t, wind_mw=w[j], q_mw=q[j], mode=nxt.mode,
                         charge_mw=log['charge_mw'], discharge_mw=log['discharge_mw'],
                         wind_dispatch_cap_mw=log['wind_dispatch_cap_mw'],
                         dispatched_wind_mw=log['dispatched_wind_mw'],
                         curtailed_wind_mw=log['curtailed_wind_mw'],
                         wind_downward_ramp_exception_mw=log['wind_downward_ramp_exception_mw'],
                         poi_mw=log['poi_mw'], tank_mwhth=nxt.energy_mwhth,
                         regulatory_action=log['regulatory_action']))
        cur = nxt
    traj = pd.DataFrame(rows)
    dev = traj.poi_mw - traj.q_mw
    return GapRolloutResult(
        issue_utc=t0, delivery_utc=t1, policy=name, wind_source=wind_source,
        initial_soc_mwhth=float(state.energy_mwhth), predicted_soc_mwhth=float(cur.energy_mwhth),
        predicted_mode_at_delivery=cur.mode, n_steps=len(times),
        gap_wind_forecast_mean_mw=float(w.mean()), anchor_wind_mw=anchor_wind_mw,
        gap_commitment_mean_mw=float(q.mean()),
        expected_shortfall_mwh=float(np.clip(-dev, 0, None).sum() * DT_H),
        expected_surplus_mwh=float(np.clip(dev, 0, None).sum() * DT_H),
        regulatory_interventions=int((traj.regulatory_action != 'none').sum()),
        curtailed_wind_mwh=float(traj.curtailed_wind_mw.sum()*DT_H),
        wind_downward_ramp_exceptions=int((traj.wind_downward_ramp_exception_mw > 1e-7).sum()),
        trajectory=traj)


def forecast_rollout(*, issue_utc, delivery_utc, state, measured_wind_5min,
                     short_term_wind_predict, anchor_wind_mw, commitments_hourly,
                     da_prices_hourly, policy='tracking_rule', relax_hours=4.0,
                     prm=PlantParams(), curtailment_enabled=False) -> GapRolloutResult:
    """Controller-side rollout: causal point wind forecast for the gap."""
    t0 = pd.Timestamp(issue_utc).tz_convert('UTC')
    meas = pd.Series(measured_wind_5min, dtype=float)
    meas.index = pd.DatetimeIndex(meas.index).tz_convert('UTC')
    if len(meas) and meas.index.max() >= t0:
        raise ValueError('wind measurements at/after the bid time are not allowed')
    n = int((pd.Timestamp(delivery_utc).tz_convert('UTC') - t0) / STEP)
    path = gap_wind_forecast(t0, n, short_term_wind_predict(t0, meas.copy()), anchor_wind_mw,
                             relax_hours=relax_hours, prm=prm)
    return rollout_delivery_soc(issue_utc=t0, delivery_utc=delivery_utc, state=state,
                                commitments_hourly=commitments_hourly,
                                da_prices_hourly=da_prices_hourly, wind_path=path,
                                wind_source=f'LL_xgb_4h_then_relax{relax_hours:g}h',
                                anchor_wind_mw=anchor_wind_mw, policy=policy, prm=prm,
                                curtailment_enabled=curtailment_enabled)


def oracle_rollout(*, issue_utc, delivery_utc, state, realised_wind_5min,
                   commitments_hourly, da_prices_hourly, policy='tracking_rule',
                   prm=PlantParams(), curtailment_enabled=False) -> GapRolloutResult:
    """POST-HOC diagnostic only: same policy model with realised gap wind.

    (forecast rollout - oracle) isolates the gap-wind forecast error;
    (oracle - realised midnight SoC) isolates the policy-model mismatch.
    """
    t0 = pd.Timestamp(issue_utc).tz_convert('UTC'); t1 = pd.Timestamp(delivery_utc).tz_convert('UTC')
    real = pd.Series(realised_wind_5min, dtype=float)
    real.index = pd.DatetimeIndex(real.index).tz_convert('UTC')
    path = real.reindex(pd.date_range(t0, t1 - STEP, freq='5min', tz='UTC')).to_numpy(float)
    return rollout_delivery_soc(issue_utc=t0, delivery_utc=t1, state=state,
                                commitments_hourly=commitments_hourly,
                                da_prices_hourly=da_prices_hourly, wind_path=path,
                                wind_source='REALISED (post-hoc oracle)', policy=policy, prm=prm,
                                curtailment_enabled=curtailment_enabled)
