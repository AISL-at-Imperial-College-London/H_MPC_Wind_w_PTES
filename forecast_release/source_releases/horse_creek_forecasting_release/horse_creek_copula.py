#!/usr/bin/env python3
"""Causal nonparametric wind/DA-price scenario fallback for Horse Creek.

Only pre-2026 data are permitted in the analogue library. Runtime history may
contain past 2026 observations, with strictly completed hourly wind / historical
DA-price rows, and is NEVER inserted into the analogue library.

Concept: paired seasonal historical 84h blocks conditioned on available recent
wind, with historical DA price ranks monotonically remapped to last 90d of
available observed DA prices. 500 raw -> weighted greedy backward 50 scenarios.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from original_backward_reduction import backward_reduce_original

TZ = 'America/Chicago'
WIND = 'HorseCreek_MW'
PRICE = 'DA_price_usd_mwh'
HORIZON = 84
N_RAW = 500
N_REDUCED = 50


def prepare_hourly(data: pd.DataFrame) -> pd.DataFrame:
    if not {WIND, PRICE}.issubset(data.columns):
        raise ValueError(f'Expected columns {WIND} and {PRICE}')
    z = data.copy()
    if 'timestamp_utc' in z.columns:
        z['timestamp_utc'] = pd.to_datetime(z['timestamp_utc'], utc=True, errors='raise', format='mixed')
        z = z.set_index('timestamp_utc')
    if not isinstance(z.index, pd.DatetimeIndex):
        raise ValueError('DatetimeIndex or timestamp_utc required')
    if z.index.tz is None:
        raise ValueError('Timezone-aware UTC input required')
    z.index = z.index.tz_convert('UTC')
    z = z.sort_index()
    if z.index.has_duplicates:
        raise ValueError('Duplicated source timestamps; resolve before running')
    z[WIND] = pd.to_numeric(z[WIND], errors='coerce')
    z[PRICE] = pd.to_numeric(z[PRICE], errors='coerce')
    if len(z) and not ((z.index.minute == 0) & (z.index.second == 0)).all():
        raise ValueError('Expected hourly interval-start timestamps only')
    # Files may have missing rows. Reindexing at runtime makes gaps explicit.
    return z[[WIND, PRICE]]


def delivery_times(delivery_date: str):
    day = pd.Timestamp(delivery_date)
    if day.tzinfo is not None:
        raise ValueError('Supply local ERCOT delivery date as YYYY-MM-DD without timezone')
    if day.strftime('%Y-%m-%d') != delivery_date:
        raise ValueError('Supply YYYY-MM-DD')
    local_start = day.tz_localize(TZ)
    tomorrow = (day + pd.DateOffset(days=1)).tz_localize(TZ)
    if tomorrow.tz_convert('UTC') - local_start.tz_convert('UTC') != pd.Timedelta(hours=24):
        raise ValueError('DST 23/25-hour local delivery day excluded from 24h design')
    prev_day = (day - pd.DateOffset(days=1)).tz_localize(TZ)
    cutoff = prev_day + pd.Timedelta(hours=9, minutes=30)
    start = local_start.tz_convert('UTC')
    return cutoff.tz_convert('UTC'), start


@dataclass
class HistoricalPool:
    starts_ns: np.ndarray
    months: np.ndarray
    winds: np.ndarray
    prices: np.ndarray
    prior_wind_mean: np.ndarray


def build_pool(data: pd.DataFrame, horizon=HORIZON) -> HistoricalPool:
    if (data.index >= pd.Timestamp('2026-01-01', tz='UTC')).any():
        raise ValueError('TRAINING/REFERENCE LEAKAGE: analogue source must end before 2026')
    starts, months, winds, prices, means = [], [], [], [], []
    full = data.index
    # Only local 00:00 starts; do not cross missing hours or historical gaps.
    for j, ts in enumerate(full):
        local = ts.tz_convert(TZ)
        if local.hour != 0 or local.minute != 0 or j < 39 or j + horizon > len(data):
            continue
        sub = data.iloc[j - 39: j + horizon]
        if len(sub) != horizon + 39 or not np.all(np.diff(sub.index.asi8) == 3600 * 10**9):
            continue
        if sub[[WIND, PRICE]].isna().any().any():
            continue
        if (ts + pd.Timedelta(hours=24)).tz_convert(TZ).hour != 0:
            continue  # DST local day
        pastwind = sub[WIND].to_numpy(dtype=float)[:24]
        winds.append(sub[WIND].to_numpy(dtype=float)[39:])
        prices.append(sub[PRICE].to_numpy(dtype=float)[39:])
        means.append(float(pastwind.mean()))
        starts.append(int(ts.value))
        months.append(int(local.month))
    if not starts:
        raise ValueError('No eligible complete historical 84h analogue blocks')
    return HistoricalPool(np.asarray(starts,dtype=np.int64),np.asarray(months,dtype=int),
                          np.stack(winds),np.stack(prices),np.array(means))


def greedy_reduce(w: np.ndarray, p: np.ndarray, keep=50):
    """Greedy backward nearest-survivor mass transfer; Euclidean robust-scaled paths.

    This is a reproducible approximation, NOT a globally optimal W1 selection.
    """
    n,h=w.shape
    if n != p.shape[0] or p.shape[1] != h or not (1 <= keep <= n):
        raise ValueError('bad scenario sizes')
    # Monotone asinh transform makes tail spikes tractable for distance metric.
    x=np.stack([w, np.arcsinh(p / 50.0)],axis=2)
    med=np.median(x, axis=(0,1))
    scale=np.quantile(x,.75,axis=(0,1))-np.quantile(x,.25,axis=(0,1))
    scale=np.maximum(scale,[15.0,0.20])
    feat=((x-med)/scale).reshape(n,-1)
    dd=cdist(feat,feat)
    np.fill_diagonal(dd,np.inf)
    active=np.ones(n,dtype=bool)
    prob=np.full(n,1/n)
    while active.sum()>keep:
        ids=np.flatnonzero(active)
        sub=dd[np.ix_(ids,ids)]
        nearest=np.argmin(sub,axis=1)
        removal_cost=prob[ids]*sub[np.arange(len(ids)),nearest]
        loser_pos=int(np.argmin(removal_cost))
        loser=int(ids[loser_pos]);survivor=int(ids[nearest[loser_pos]])
        prob[survivor]+=prob[loser]
        prob[loser]=0
        active[loser]=False
    kept=np.flatnonzero(active)
    ww=prob[kept]
    assert len(kept)==keep and np.isclose(ww.sum(),1)
    return kept,ww


@dataclass
class ScenarioBundle:
    delivery_day: str
    cutoff_utc: str
    timestamps_utc: pd.DatetimeIndex
    wind_mw: np.ndarray              # (n_keep,84)
    da_price_usd_mwh: np.ndarray     # (n_keep,84)
    weights: np.ndarray              # (n_keep,)
    raw_wind_mw: np.ndarray          # (n_raw,84)
    raw_da_price_usd_mwh: np.ndarray # (n_raw,84)
    metadata: dict

    def as_npz(self,path):
        np.savez_compressed(path, wind_mw=self.wind_mw,
                            da_price_usd_mwh=self.da_price_usd_mwh,
                            probabilities=self.weights,
                            delivery_timestamps_utc=np.asarray(self.timestamps_utc.strftime('%Y-%m-%dT%H:%M:%SZ'), dtype='U24'))

    def as_csv(self,path):
        h=self.wind_mw.shape[1]
        n=self.wind_mw.shape[0]
        return pd.DataFrame({
            'scenario_id': np.repeat(np.arange(n),h),
            'probability': np.repeat(self.weights,h),
            'delivery_timestamp_utc': np.tile(self.timestamps_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),n),
            'wind_MW': self.wind_mw.ravel(),
            'DA_price_USD_MWh': self.da_price_usd_mwh.ravel()
        }).to_csv(path,index=False)


class CopulaFallback:
    """Historical rank copula with causal price-marginal recalibration.

    2024 and 2025: chronological development. For 2026, runtime_history
    REQUIRED (actual pre-cutoff hourly observations). Frozen analogue library
    remains 2018–2025, NEVER refitted from 2026 deployment observations.
    """
    def __init__(self, historical_development_data: pd.DataFrame):
        self.dev=prepare_hourly(historical_development_data)
        self.pool=build_pool(self.dev)

    @classmethod
    def from_csv(cls,path):
        return cls(pd.read_csv(path))

    def generate(self, delivery_day: str, runtime_history: Optional[pd.DataFrame]=None,
                 n_raw=N_RAW, n_keep=N_REDUCED, seed=3714) -> ScenarioBundle:
        cutoff,start=delivery_times(delivery_day)
        if n_raw<2 or n_keep>n_raw or n_keep<1:
            raise ValueError('n_raw >= n_keep >=1 required')
        if cutoff.year>=2026 and runtime_history is None:
            raise ValueError('2026 deployment requires supplied causal runtime history; holdout not read automatically')
        now=self.dev.loc[self.dev.index+pd.Timedelta(hours=1)<=cutoff].copy()
        if runtime_history is not None:
            observed=prepare_hourly(runtime_history)
            if (observed.index+pd.Timedelta(hours=1)>cutoff).any():
                raise ValueError('RUNTIME LOOKAHEAD: history contains incomplete or future hour')
            # Historical source in reference file is immutable; runtime history overrides
            # only observed values with valid provenance supplied by caller.
            now=pd.concat([now,observed]).sort_index()
            now=now[~now.index.duplicated(keep='last')]
        if now.empty:
            raise ValueError('No prior history')
        prev_full_hour=cutoff.floor('h')-pd.Timedelta(hours=1)
        hours=pd.date_range(end=prev_full_hour,periods=24,freq='h',tz='UTC')
        recent_w=now.reindex(hours)[WIND]
        if recent_w.isna().any():
            raise ValueError('Not enough complete, contiguous measured wind history at cutoff')
        wind_mean=float(recent_w.mean())
        hist_da=now[PRICE].dropna()
        recent_da=hist_da.loc[(hist_da.index>=cutoff-pd.Timedelta(days=90)) &
                              (hist_da.index+pd.Timedelta(hours=1)<=cutoff)]
        if len(recent_da)<45*24 or (cutoff-(recent_da.index.max()+pd.Timedelta(hours=1)))>pd.Timedelta(hours=24):
            raise ValueError('Missing or stale recent DA history: need >=45 days in past 90 and last <=24h')
        month=start.tz_convert(TZ).month
        md=np.abs(self.pool.months-month)
        sel=(np.minimum(md,12-md)<=2) & \
            (self.pool.starts_ns+(HORIZON*3600*10**9)<=cutoff.value)
        idx=np.flatnonzero(sel)
        if len(idx)<40:
            raise ValueError(f'Only {len(idx)} eligible historical analogues (<40)')
        dist=np.abs(self.pool.prior_wind_mean[idx]-wind_mean)
        ww=0.15+np.exp(-0.5*(dist/50.)**2)
        ww/=ww.sum()
        rng=np.random.default_rng(seed)
        draw=rng.choice(idx,size=n_raw,replace=True,p=ww)
        w=self.pool.winds[draw].copy()
        rawprice=self.pool.prices[draw]
        source_sorted=np.sort(self.pool.prices[idx].reshape(-1))
        target_sorted=np.sort(recent_da.to_numpy(dtype=float))
        ranks=np.searchsorted(source_sorted,rawprice,side='right')/(len(source_sorted)+1)
        xp=(np.arange(len(target_sorted))+0.5)/len(target_sorted)
        p=np.interp(np.clip(ranks,xp[0],xp[-1]),xp,target_sorted)
        if not (np.isfinite(w).all() and np.isfinite(p).all() and (w>=-1e-8).all() and (w<=230.001).all()):
            raise ValueError('Scenarios contain invalid data or wind outside rated range')
        keep,probs=backward_reduce_original(w,p,n_keep)
        meta={
            'method':'causal_historical_empirical_copula_recent_DA_quantile_transport',
            'delivery_day':delivery_day,'cutoff_utc':cutoff.isoformat(),
            'n_raw':n_raw,'n_keep':n_keep,'horizon_h':HORIZON,'seed':int(seed),
            'candidate_count':int(len(idx)),
            'max_analogue_end_utc':pd.Timestamp(self.pool.starts_ns[idx].max()+HORIZON*3600*10**9,unit='ns',tz='UTC').isoformat(),
            'last_observed_wind_hour_start_utc':hours[-1].isoformat(),
            'last_observed_DA_price_hour_start_utc':recent_da.index.max().isoformat(),
            'recent_DA_observations':len(recent_da),
            'recent_DA_window_days':90,'wind_analogue_kernel_MW':50,
            'minimum_month_matched_analogue_blocks':40,
            'empirical_copula_reference_library_max_year':2025,
            'scenario_reduction':'original_algorithm_2_backward_deleted_mass_W1_Euclidean_standardized_joint_HL84',
            'scenario_probability_sum':float(probs.sum()),
            'effective_scenario_count':float(1/np.sum(probs**2)),
            'future_realizations_accessed':False,
        }
        return ScenarioBundle(delivery_day,cutoff.isoformat(),pd.date_range(start,periods=HORIZON,freq='h'),
                              w[keep],p[keep],probs,w,p,meta)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default=str(Path(__file__).resolve().parent/'data'/'development_master_hourly_2018_2021_2024_2025.csv'))
    parser.add_argument('--day',default='2025-07-14',help='Development date only by default (2026 requires --runtime-history)')
    parser.add_argument('--runtime-history',default=None,help='CSV containing ONLY completely observed hourly rows before cutoff')
    parser.add_argument('--n-raw',type=int,default=500)
    parser.add_argument('--n-keep',type=int,default=50)
    parser.add_argument('--seed',type=int,default=3714)
    parser.add_argument('--output-dir',default=str(Path(__file__).resolve().parent/'results'))
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    gen=CopulaFallback.from_csv(args.data)
    runtime=pd.read_csv(args.runtime_history) if args.runtime_history else None
    s=gen.generate(args.day,runtime_history=runtime,n_raw=args.n_raw,n_keep=args.n_keep,seed=args.seed)
    basename=f'copula_{args.day}_{args.n_raw}to{args.n_keep}'
    s.as_npz(out/f'{basename}.npz')
    s.as_csv(out/f'{basename}.csv')
    (out/f'{basename}_metadata.json').write_text(json.dumps(s.metadata,indent=2))
    print(json.dumps(s.metadata,indent=2))
    print('Wrote',out/f'{basename}.npz')


if __name__=='__main__':main()
