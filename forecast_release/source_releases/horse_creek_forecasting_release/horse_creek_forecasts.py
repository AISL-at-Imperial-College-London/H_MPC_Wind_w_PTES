#!/usr/bin/env python3
"""Causal forecast provider API for the Horse Creek / ERCOT study.

Modified 2026-10-17: TCVAE method uses continuous 120h joint CVAE v3
with fixed 2025 March-June trajectory calibration. Backward reduction is
computed only on the 84h HL decision-relevant slice. Legacy checkpoints remain
for reproducibility. No 2026 training or calibration data are opened.

This package does not open 2026 deployment files. The caller supplies observations
that were genuinely available at the requested decision time. No MPC in this file.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
import math
from typing import Optional
import numpy as np
import pandas as pd

from horse_creek_copula import CopulaFallback, ScenarioBundle, delivery_times, prepare_hourly
from original_backward_reduction import backward_reduce_original

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/development_master_hourly_2018_2021_2024_2025.csv'
MODEL=ROOT/'models/continuous_cvae_v3_checkpoint.pt'
CALIBRATION=ROOT/'models/continuous_calibration_residual_paths.npz'
MODEL_LEGACY=ROOT/'models/thesis_tcvae_epoch250_checkpoint.pt'
MODEL_BEST=ROOT/'models/thesis_tcvae_best_validation_checkpoint.pt'
RATING=230.0


def _utc(ts):
    x=pd.Timestamp(ts)
    if x.tzinfo is None:
        raise ValueError('Decision timestamp must be timezone-aware')
    return x.tz_convert('UTC')


def _series(src, column, *, cadence=None):
    if isinstance(src, pd.Series):
        s=src.copy()
    elif isinstance(src, pd.DataFrame):
        if column not in src: raise ValueError(f'Missing {column}')
        if 'timestamp_utc' in src:
            s=src.set_index('timestamp_utc')[column].copy()
        else: s=src[column].copy()
    else: raise TypeError('Supply pandas Series or DataFrame')
    if not isinstance(s.index,pd.DatetimeIndex):s.index=pd.to_datetime(s.index,utc=True)
    if s.index.tz is None: raise ValueError('Timestamps must have timezone offsets; use UTC')
    s.index=s.index.tz_convert('UTC')
    if s.index.has_duplicates:raise ValueError('Duplicate timestamps')
    if cadence is not None and not ((s.index.asi8%(cadence*60*10**9))==0).all():
        raise ValueError(f'Non-{cadence}min interval start timestamps')
    s=pd.to_numeric(s,errors='coerce').sort_index()
    return s


@dataclass
class PointForecast:
    issue_time_utc: pd.Timestamp
    target_times_utc: pd.DatetimeIndex
    values: np.ndarray
    variable: str
    units: str
    metadata: dict

    def to_frame(self):
        return pd.DataFrame({'timestamp_utc':self.target_times_utc, self.variable:self.values})


def wind_5min(issue_time_utc, observed_5min, *, max_measurement_age_minutes=15):
    """48 x 5min frozen persistence selected on January 2026 DEV validation.

    For operational correctness, only observations from strictly before issue
    time (t-5 at the latest) are allowed, even if callers use a current sample.
    """
    t=_utc(issue_time_utc)
    if t.minute%15 or t.second or t.microsecond:raise ValueError('LL solve must be at a 15min boundary')
    s=_series(observed_5min,'HorseCreek_MW',cadence=5)
    if s.empty:raise ValueError('Missing wind history')
    if s.index.max()>=t:raise ValueError('Wind lookahead: measurements at/after issue time forbidden')
    good=s.dropna()
    if good.empty:raise ValueError('No measured wind values')
    last=good.index[-1]
    if last>t-pd.Timedelta(minutes=5):raise ValueError('Future wind observation')
    age=(t-last).total_seconds()/60
    if age>max_measurement_age_minutes:raise ValueError(f'Wind measurement stale by {age:.1f} minutes')
    value=float(good.iloc[-1])
    if not (0<=value<=RATING):raise ValueError('Wind observation outside physical range')
    return PointForecast(t,pd.date_range(t,periods=48,freq='5min',tz='UTC'),np.full(48,value),
                         'wind_mw','MW',dict(model='5min_persistence_frozen',last_measured_utc=last.isoformat(),
                                             information_delay_min=age,selected_using='2026-01-21..28 development',
                                             n_steps=48,horizon_minutes=240))


def rt15(issue_time_utc, cleared_da_hourly, *, da_asof_utc=None, model='DA_only',
         observed_rt15=None, publication_lag_minutes=60, allow_missing_rt=False):
    """16 x 15-minute ERCOT RT price point forecast.

    PRIMARY model: DA_only, selected using 2025 predeployment development
    because it beat RT persistence on the 4h horizon and requires no speculative
    RT price disclosure timing. Future DA *clearing* must be independently
    verified by caller at issue time; DA values in a final CSV alone do NOT
    establish availability.

    SENSITIVITY: RT_persistence selected originally on 2021 validation.
    For RT_persistence, caller must provide only prices proven published by
    issue time. publication_lag_minutes is a conservative *assumption*
    after settlement interval END, not a verified market-data timestamp.
    """
    t=_utc(issue_time_utc)
    if t.minute%15 or t.second or t.microsecond:raise ValueError('RT forecast issues on 15min boundaries')
    if da_asof_utc is None:raise ValueError('Must attest DA clearance time: da_asof_utc required')
    published=_utc(da_asof_utc)
    if published>t:raise ValueError('DA price as-of timestamp after forecast issue time')
    if model not in ('DA_only','RT_persistence'):raise ValueError('Unknown RT forecasting method')
    if publication_lag_minutes<0: raise ValueError('Invalid publication delay')
    da=_series(cleared_da_hourly,'DA_price_usd_mwh',cadence=60)
    fut=pd.date_range(t,periods=16,freq='15min',tz='UTC')
    target_hour=fut.floor('h')
    DA=da.reindex(target_hour)
    if DA.isna().any():raise ValueError('Missing CLEARED DA prices. Supply available values with publication-time provenance.')
    source=model;last=None
    if model=='DA_only':
        prices=DA.to_numpy(float)
    else:
        if observed_rt15 is None:raise ValueError('RT persistence requires historical published RT records')
        rt=_series(observed_rt15,'RT15_price_usd_mwh',cadence=15)
        eligible_latest=t-pd.Timedelta(minutes=15+publication_lag_minutes)
        if len(rt) and rt.index.max()>eligible_latest:
            raise ValueError(f'RT record not eligible at issue time under delay. Latest allowable start {eligible_latest}')
        good=rt.dropna()
        if good.empty or t-good.index.max()>pd.Timedelta(hours=12):
            if not allow_missing_rt:raise ValueError('No eligible recent RT prices')
            source='DA_only_missing_RT_fallback';prices=DA.to_numpy(float)
        else:
            last=good.index[-1]
            prices=np.full(16,float(good.iloc[-1]))
    return PointForecast(t,fut,prices,'rt15_price_usd_mwh','USD/MWh',
                         dict(model=source, primary_frozen_using='2025_development',
                              RT_persistence_2021_validation_sensitivity_only=True,
                              publication_lag_minutes=int(publication_lag_minutes) if model=='RT_persistence' else None,
                              RT_publication_delay_is_unverified_assumption=True,
                              last_RT_start_utc=last.isoformat() if last is not None else None,
                              DA_asof_utc=published.isoformat(),DA_clearing_asof_provenance_required_from_caller=True,n_intervals=16))


def _join_causal_history(day,dev,runtime_history):
    cutoff,start=delivery_times(day)
    known=prepare_hourly(dev).loc[lambda z:z.index+pd.Timedelta(hours=1)<=cutoff]
    if runtime_history is not None:
        o=prepare_hourly(runtime_history)
        if len(o) and (o.index+pd.Timedelta(hours=1)>cutoff).any():
            raise ValueError('Runtime 2026 data contain incomplete or future hourly records')
        known=pd.concat([known,o]).sort_index()
        known=known.loc[~known.index.duplicated(keep='last')]
    elif cutoff.year>=2026:
        raise ValueError('2026 forecast requires explicit as-of runtime history')
    return cutoff,start,known


class JointForecastProvider:
    """MPC-compatible 84h slices from continuous 120h joint wind and DA price scenarios.

    method='tcvae' is the NEW continuous CVAE v3 + pre-2025-Jul
    trajectory calibration. All 2026 as-of history safeguards are retained.
    method='copula' preserves the independent historical copula baseline.
    method='tcvae_legacy' is the 250-epoch thesis CVAE for scientific checks.
    """
    def __init__(self,method='tcvae',development_csv=DATA,checkpoint=None):
        if method not in ('tcvae','copula','tcvae_legacy'):
            raise ValueError('Only tcvae, copula, tcvae_legacy')
        self.method=method
        self.source=Path(development_csv)
        self.dev=prepare_hourly(pd.read_csv(self.source))
        self.checkpoint=Path(checkpoint) if checkpoint is not None else (MODEL_LEGACY if method=='tcvae_legacy' else MODEL)
        self._model=None;self._scaling=None;self._cp=None
        self._continuous_provider=None
        self._copula=CopulaFallback(self.dev) if method=='copula' else None

    def _initialize_model(self):
        if self.method=='tcvae':
            if self._continuous_provider is None:
                from continuous_cvae_provider import ContinuousCalibratedCVAE
                self._continuous_provider=ContinuousCalibratedCVAE(self.checkpoint,CALIBRATION)
                self._cp=self._continuous_provider.cp
            return
        if self._model is not None:return
        import torch
        from train_thesis_tcvae import ThesisTCVAE
        cp=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
        if cp.get('history_h')!=96 or cp.get('future_h')!=96:
            raise ValueError('Expected legacy thesis T-CVAE 96/96 model')
        scale=cp['scaling']
        if min(scale['std'])<=0:raise ValueError('Invalid legacy scaling')
        model=ThesisTCVAE(hidden=int(cp['hidden']),latent_dim=int(cp['latent_dim']))
        model.load_state_dict(cp['model'],strict=True);model.eval()
        self._model=model;self._scaling=scale;self._cp=cp

    def generate(self,delivery_day,*,runtime_history=None,n_raw=500,n_keep=50,seed=3714)->ScenarioBundle:
        cutoff,start,known=_join_causal_history(delivery_day,self.dev,runtime_history)
        if self.method=='copula':
            return self._copula.generate(delivery_day,runtime_history=runtime_history,
                                         n_raw=n_raw,n_keep=n_keep,seed=seed)
        if not (2<=n_raw<=10000 and 1<=n_keep<=n_raw):raise ValueError('Bad scenario count')
        self._initialize_model()
        last=cutoff.floor('h')-pd.Timedelta(hours=1)
        histidx=pd.date_range(end=last,periods=96,freq='h',tz='UTC')
        historical=known.reindex(histidx)[['HorseCreek_MW','DA_price_usd_mwh']]
        if historical.isna().any().any():
            bad=historical.index[historical.isna().any(axis=1)][0]
            raise ValueError(f'Missing completed hourly history at {bad}')
        first_unobserved=last+pd.Timedelta(hours=1)
        gap_hours=(start-first_unobserved)/pd.Timedelta(hours=1)
        if gap_hours!=int(gap_hours) or gap_hours<0:
            raise ValueError('HL start must be an hourly timestamp after the last measurement')
        lag=int(gap_hours)
        target=pd.date_range(start,periods=84,freq='h',tz='UTC')
        # Forecast exactly until the end of the HL horizon. No artificial jump
        # at delivery: the first forecast hour is immediately after available history.
        full_h=lag+len(target)
        if full_h>120:raise ValueError(f'Forecast requires {full_h}h; trained model has 120h')
        if self.method=='tcvae':
            full_w_120,full_p_120=self._continuous_provider.generate_raw(
                historical.to_numpy(float),first_unobserved,n=n_raw,seed=seed)
            full_w=full_w_120[:,:full_h].copy()
            full_p=full_p_120[:,:full_h].copy()
            # The reduction sees ONLY the HL marginal (all 84 relevant hours).
            ww=full_w[:,lag:lag+84]
            pp=full_p[:,lag:lag+84]
            label='continuous_CVAE_v3_calibrated'
            cp_epoch=int(self._cp.get('best_epoch',-1))
            train='2018-2021 and 2024';val='2025 Jan-Feb';cal='2025 Mar-Jun'
        else:
            # Historical reference for controlled comparisons (not default).
            import torch
            dow=start.tz_convert('America/Chicago').dayofweek
            hour=start.tz_convert('America/Chicago').hour
            scale=self._scaling
            raw=historical.to_numpy(float).copy()
            if scale['price_transform']=='asinh':raw[:,1]=np.arcsinh(raw[:,1]/50)
            z=((raw-np.asarray(scale['mean']))/np.asarray(scale['std'])).astype(np.float32)
            with torch.inference_mode():
                draw=self._model.sample(torch.from_numpy(z[None,:,:]),
                    torch.tensor([dow]),torch.tensor([hour]),n=n_raw,seed=seed)
            out=draw.astype('float64')*np.asarray(scale['std'])+np.asarray(scale['mean'])
            if scale['price_transform']=='asinh':out[:,:,1]=50*np.sinh(np.clip(out[:,:,1],-9,9))
            ww=np.clip(out[:,:84,0],0,RATING);pp=out[:,:84,1]
            label='legacy_thesis_CVAE_epoch250';cp_epoch=int(self._cp.get('epoch',-1))
            train='2018-2021';val='2025 Jan-Jun';cal='none'
        if not (np.isfinite(ww).all() and np.isfinite(pp).all()):raise RuntimeError('Nonfinite CVAE outputs')
        # For historical thesis CVAE comparisons only, its first 84h are
        # delivery-aligned; no continuous precursor was trained for that legacy model.
        if self.method!='tcvae':
            full_w=None;full_p=None;first_unobserved=None;full_h=84;lag=0
        picked,probs=backward_reduce_original(ww,pp,n_keep)
        import hashlib
        digest=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        meta={'method':label,'delivery_day':delivery_day,'cutoff_utc':cutoff.isoformat(),
              'history_latest_interval_start_utc':last.isoformat(),
              'n_raw':int(n_raw),'n_keep':int(n_keep),'horizon_h':84,'seed':seed,
              'model_checkpoint_sha256':digest,'checkpoint_epoch':cp_epoch,
              'model_training_years':train,'model_validation_window':val,
              'model_calibration_window':cal,'no_2026_training_or_calibration':True,
              'calibration_sha256':hashlib.sha256(CALIBRATION.read_bytes()).hexdigest() if self.method=='tcvae' else None,
              'model_architecture':('96 completed history + continuous calibrated 120h conditional VAE' if self.method=='tcvae' else 'legacy thesis CVAE'),
              'first_unobserved_hour_utc':first_unobserved.isoformat() if self.method=='tcvae' else None,
              'forecast_full_hours':int(full_h),'pre_delivery_forecast_hours':int(lag),
              'full_forecast_end_exclusive_utc':(first_unobserved+pd.Timedelta(hours=full_h)).isoformat() if self.method=='tcvae' else None,
              'reduction_input_hours':84,'reduction_uses_pre_delivery_hours':False,
              'scenario_reduction':'original_algorithm_2_backward_deleted_mass_W1_Euclidean_standardized_joint_HL84',
              'scenario_probability_sum':float(probs.sum()),
              'effective_scenario_count':float(1/np.sum(probs**2)),
              'wind_physical_clipping':True,
              'price_tail_guard_usd_mwh':[-1000,10000] if self.method=='tcvae' else None}
        bundle=ScenarioBundle(delivery_day,cutoff.isoformat(),target,
                              ww[picked],pp[picked],probs,ww,pp,meta)
        # Diagnostic-only full trajectories. The MPC interface above is unchanged.
        if self.method=='tcvae':
            bundle.full_timestamps_utc=pd.date_range(first_unobserved,periods=full_h,freq='h',tz='UTC')
            bundle.full_raw_wind_mw=full_w
            bundle.full_raw_da_price_usd_mwh=full_p
            bundle.full_reduced_wind_mw=full_w[picked]
            bundle.full_reduced_da_price_usd_mwh=full_p[picked]
            bundle.reduced_indices=picked
        return bundle

def doctor():
    from importlib import import_module
    for name in ['numpy','pandas','scipy','torch','sklearn']:
        x=import_module(name);print(name,getattr(x,'__version__','installed'))
    print('DATA',DATA.exists(),'T-CVAE checkpoint',MODEL.exists())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('doctor')
    p=sp.add_parser('joint');p.add_argument('--day',default='2025-07-14');p.add_argument('--provider',choices=['tcvae','copula'],default='tcvae')
    p.add_argument('--runtime-history');p.add_argument('--raw',type=int,default=500);p.add_argument('--keep',type=int,default=50)
    p.add_argument('--seed',type=int,default=3714);p.add_argument('--out',default='results')
    args=ap.parse_args()
    if args.cmd=='doctor':doctor();return
    g=JointForecastProvider(args.provider)
    runtime=pd.read_csv(args.runtime_history) if args.runtime_history else None
    s=g.generate(args.day,runtime_history=runtime,n_raw=args.raw,n_keep=args.keep,seed=args.seed)
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    stem=f'{args.provider}_{args.day}_{args.raw}to{args.keep}'
    s.as_npz(out/f'{stem}.npz');s.as_csv(out/f'{stem}.csv')
    (out/f'{stem}.json').write_text(json.dumps(s.metadata,indent=2))
    print('Created',out/f'{stem}.npz')
    print(json.dumps(s.metadata,indent=2))

if __name__=='__main__':main()
