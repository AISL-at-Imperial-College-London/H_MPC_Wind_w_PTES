"""Deterministic HL baseline for the *frozen* Horse Creek continuous-CVAE v3 package.

Only at the high-level optimization boundary, map the 50 *weighted* joint CVAE
scenarios to their one probability-weighted hourly mean path. Keep the full
500->50 forecasting/reduction, causal SoC rollout, lower-level 5-min MIQP,
physical replay, and two-settlement financial accounting unchanged.

Do NOT monkeypatch the generator. The original CVAE ensemble must still be
available to the rollout and for the reported forecast mean in both policies.
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

FORECAST_SCENARIOS = 50
HORIZON_HOURS = 84


def mean_hl_inputs(wind_scenarios, da_scenarios, weights, *, expected_scenarios=FORECAST_SCENARIOS):
    """Return (wind[1,84], price[1,84], weights[1]) without rounding.

    This is a JOINT weighted mean trajectory: paired wind and price rows have
    the same weights, so there is no independent mean, median or resampling.
    """
    w = np.asarray(wind_scenarios, dtype=np.float64)
    p = np.asarray(da_scenarios, dtype=np.float64)
    pi = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (expected_scenarios, HORIZON_HOURS) or p.shape != w.shape:
        raise AssertionError(f'Expected {(expected_scenarios, HORIZON_HOURS)} joint forecasts, got {w.shape}, {p.shape}')
    if pi.shape != (expected_scenarios,) or np.any(pi <= 0) or not np.isclose(pi.sum(), 1., atol=1e-10, rtol=0):
        raise AssertionError('Invalid reduced-scenario probabilities')
    if not (np.isfinite(w).all() and np.isfinite(p).all() and np.isfinite(pi).all()):
        raise AssertionError('Nonfinite ensemble')
    if np.any(w < -1e-8) or np.any(w > 230. + 1e-8):
        raise AssertionError('Wind outside farm rating')
    wmean = pi @ w
    pmean = pi @ p
    if np.any(wmean < -1e-8) or np.any(wmean > 230. + 1e-8):
        raise AssertionError('Invalid deterministic wind')
    return wmean.reshape(1, HORIZON_HOURS), pmean.reshape(1, HORIZON_HOURS), np.ones(1, dtype=np.float64)


def ensemble_sha256(wind, price, weights):
    h = hashlib.sha256()
    for obj in (wind, price, weights):
        a = np.ascontiguousarray(obj, dtype='<f8')
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class CaseLogger:
    dest: Path
    first_market_day: str
    case_name: str
    count: int = 0

    def record(self, wind, price, weights):
        w, p, pr = mean_hl_inputs(wind, price, weights)
        self.count += 1
        if self.count > 15:
            raise AssertionError('More than 15 daily HL issues for a 15-day local episode')
        day = (pd.Timestamp(self.first_market_day) + pd.Timedelta(days=self.count-1)).strftime('%Y-%m-%d')
        midnight = pd.Timestamp(day).tz_localize('America/Chicago').tz_convert('UTC')
        issue = midnight-pd.Timedelta(hours=14, minutes=30)
        entries=[]
        for h in range(HORIZON_HOURS):
            entries.append(dict(hl_issue_index=self.count,
                                delivery_day_local=day, forecast_issue_utc=issue.isoformat(),
                                hour_start_utc=(midnight+pd.Timedelta(hours=h)).isoformat(),
                                deterministic_wind_mean_mw=float(w[0,h]),
                                deterministic_DA_price_mean_usd_mwh=float(p[0,h]),
                                forecast_retained_scenarios=FORECAST_SCENARIOS,
                                hl_optimization_scenarios=1))
        self.dest.mkdir(parents=True, exist_ok=True)
        hourly = self.dest/'DETERMINISTIC_HL_MEAN_INPUTS.csv'
        self._append(hourly, entries)
        audit = dict(hl_issue_index=self.count, delivery_day_local=day,
                     forecast_issue_utc=issue.isoformat(),
                     case=self.case_name, forecast_retained_scenarios=FORECAST_SCENARIOS,
                     hl_optimization_scenarios=1,
                     retained_probability_sum=float(np.sum(weights)),
                     reduced_ensemble_sha256=ensemble_sha256(wind,price,weights),
                     expected_delivery_wind_mwh=float(w[0,:24].sum()),
                     mean_delivery_DA_price_usd_mwh=float(p[0,:24].mean()))
        self._append(self.dest/'DETERMINISTIC_HL_INPUT_AUDIT.csv',[audit])
        return w,p,pr

    @staticmethod
    def _append(path, rows):
        if not rows:return
        exists = path.exists() and path.stat().st_size > 0
        with path.open('a', newline='', encoding='utf-8') as f:
            wr=csv.DictWriter(f, fieldnames=list(rows[0]))
            if not exists:wr.writeheader()
            wr.writerows(rows)
