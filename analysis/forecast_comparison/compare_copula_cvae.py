#!/usr/bin/env python3
"""Paired causal scenario forecast comparison: frozen CVAE vs shipped copula.

Runs both existing forecast providers, without retraining or modifying their implementations.
Test issues D-1 at 09:30 US Central; evaluation starts at midnight local on D,
using every hour of the 84-hour high-level MPC forecast.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2]/'forecast_release').exists() else Path(__file__).parent/'horse-creek-ptes-mpc'
WEEKS = [
    ('jul_2025','2025-07-14','2025-07-20'),
    ('sep_2025','2025-09-08','2025-09-14'),
    ('nov_2025','2025-11-10','2025-11-16'),
    ('dec_2025','2025-12-08','2025-12-14'),
    ('feb_2026_84h','2026-02-09','2026-02-12'),
    ('feb_2026_24h','2026-02-09','2026-02-14'),
]

def weighted_quantile(x,w,q):
    ii=np.argsort(x,axis=0)
    xx=np.take_along_axis(x,ii,axis=0)
    ww=np.take_along_axis(np.broadcast_to(w[:,None],x.shape),ii,axis=0)
    return np.array([xx[np.minimum(np.argmax(np.cumsum(ww,axis=0)>=v,axis=0),len(x)-1),np.arange(x.shape[1])] for v in q])

def crps_ensemble(x,w,y):
    # exact empirical weighted CRPS without n^2 memory:
    # E|X-y| - .5 E|X-X'|, with X and X' independently drawn with weights w.
    first=np.sum(w[:,None]*np.abs(x-y[None,:]),axis=0)
    order=np.argsort(x,axis=0)
    xs=np.take_along_axis(x,order,axis=0)
    ws=np.take_along_axis(np.broadcast_to(w[:,None],x.shape),order,axis=0)
    cw=np.cumsum(ws,axis=0)-ws
    cs=np.cumsum(ws*xs,axis=0)-ws*xs
    pair=np.sum(ws*(xs*cw-cs),axis=0)
    return first-pair

def energy_trajectory(wind,price,prob,truth_w,truth_p):
    # Transform both channels to dimensionless coordinates; keep temporal pairing.
    xx=np.column_stack([wind/230.0,np.arcsinh(price/50.0)])
    yy=np.r_[truth_w/230.0,np.arcsinh(truth_p/50.0)]
    from scipy.spatial.distance import cdist
    diff=cdist(xx,yy[None,:]).ravel()
    between=cdist(xx,xx)
    return float(prob @ diff - 0.5 * prob @ between @ prob)

def score_group(x,w,y):
    mean=np.sum(w[:,None]*x,axis=0)
    qq=weighted_quantile(x,w,[.05,.1,.25,.75,.9,.95])
    lower90,lower80,lower50,upper50,upper80,upper90=qq
    return dict(crps=crps_ensemble(x,w,y),ae=np.abs(mean-y),squared=(mean-y)**2,
                cover50=((y>=lower50)&(y<=upper50)).astype(float),
                cover80=((y>=lower80)&(y<=upper80)).astype(float),
                cover90=((y>=lower90)&(y<=upper90)).astype(float),
                width50=upper50-lower50,width80=upper80-lower80,width90=upper90-lower90)

def row_spearman(a,b):
    """Spearman rank correlations over time for each scenario (vectorized)."""
    from scipy.stats import rankdata
    x=rankdata(a,axis=1);y=rankdata(b,axis=1)
    x=x-x.mean(axis=1,keepdims=True);y=y-y.mean(axis=1,keepdims=True)
    denom=np.sqrt(np.sum(x*x,axis=1)*np.sum(y*y,axis=1))
    return np.divide(np.sum(x*y,axis=1),denom,out=np.zeros(len(a)),where=denom>0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--out',type=Path,default=Path.cwd()/'forecast_comparison_results')
    ap.add_argument('--weeks',nargs='*',default=[w[0] for w in WEEKS if '2025' in w[0]])
    ap.add_argument('--raw',type=int,default=500)
    ap.add_argument('--keep',type=int,default=50)
    ap.add_argument('--seed',type=int,default=3714)
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    provider_path=args.root.resolve()/'forecast_release/source_releases/horse_creek_forecasting_release'
    sys.path.insert(0,str(provider_path))
    from horse_creek_forecasts import JointForecastProvider,DATA,MODEL,CALIBRATION
    from horse_creek_copula import delivery_times, prepare_hourly
    from scipy.stats import spearmanr,rankdata
    recorded={}
    for label,p in [('development_master',DATA),('frozen_cvae',MODEL),('calibration',CALIBRATION)]:
        recorded[label]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    reference=prepare_hourly(pd.read_csv(DATA))
    winter_csv=args.root.resolve()/'holdout_inputs/winter/HL_realized_hourly_wind_DA.csv'
    winter=pd.read_csv(winter_csv).rename(columns={'HorseCreek_MW_hourly':'HorseCreek_MW'})[['timestamp_utc','HorseCreek_MW','DA_price_usd_mwh']]
    winter=prepare_hourly(winter)
    models={m:JointForecastProvider(m) for m in ['tcvae','copula']}
    daily=[];hourly=[];issuance=[];summary=[]
    for week,start,end in WEEKS:
        if week not in args.weeks: continue
        for t in pd.date_range(start,end,freq='D'):
            day=t.strftime('%Y-%m-%d')
            cutoff,delivery=delivery_times(day)
            scoring_hours=24 if week=='feb_2026_24h' else 84
            truth_dates=pd.date_range(delivery,periods=scoring_hours,freq='h',tz='UTC')
            actual=(winter if '2026' in week else reference).reindex(truth_dates)
            if actual.isna().any().any():
                raise RuntimeError(f'Unexpected gap in {day} truth; no silent imputation')
            realw=actual['HorseCreek_MW'].to_numpy(float)
            realp=actual['DA_price_usd_mwh'].to_numpy(float)
            if not (np.isfinite(realw).all() and np.isfinite(realp).all()):
                raise RuntimeError(f'Missing truth for {day}')
            for method,provider in models.items():
                started=time.perf_counter()
                runtime=None
                if '2026' in week:
                    runtime=winter.loc[winter.index+pd.Timedelta(hours=1)<=cutoff].copy()
                b=provider.generate(day,runtime_history=runtime,n_raw=args.raw,n_keep=args.keep,seed=args.seed)
                elapsed=time.perf_counter()-started
                if not (b.timestamps_utc[:scoring_hours].equals(truth_dates) and np.isclose(b.weights.sum(),1)):
                    raise ValueError('Scenario timestamps/weights invalid')
                if b.metadata['cutoff_utc'] != cutoff.isoformat():
                    raise ValueError('Issue time mismatch')
                issuance.append(dict(week=week,day=day,method=method,cutoff_utc=cutoff.isoformat(),
                                     scenario_count_raw=args.raw,scenario_count_reduced=args.keep,
                                     duration_seconds=round(elapsed,3),
                                     effective_reduced_scenarios=b.metadata.get('effective_scenario_count'),
                                     meta=b.metadata))
                print(week, day, method,'%.1fs'%elapsed,flush=True)
                for ensemble,wind,price,wt in [(f'raw_{args.raw}',b.raw_wind_mw[:,:scoring_hours],b.raw_da_price_usd_mwh[:,:scoring_hours],np.ones(args.raw)/args.raw),
                                               (f'reduced_{args.keep}',b.wind_mw[:,:scoring_hours],b.da_price_usd_mwh[:,:scoring_hours],b.weights)]:
                    score_w=score_group(wind,wt,realw)
                    score_p=score_group(price,wt,realp)
                    for h in range(scoring_hours):
                        row=dict(week=week,day=day,method=method,ensemble=ensemble,
                                 lead_hour=int(h),group='first24' if h<24 else ('next24' if h<48 else 'last36'),
                                 actual_wind_MW=realw[h],actual_DA_USD_MWh=realp[h])
                        for var,z in [('wind',score_w),('price',score_p)]:
                            for key,value in z.items():row[f'{var}_{key}']=float(value[h])
                        hourly.append(row)
                    daily.append(dict(week=week,day=day,method=method,ensemble=ensemble,
                                      wind_crps=np.mean(score_w['crps']),price_crps=np.mean(score_p['crps']),
                                      wind_mae=np.mean(score_w['ae']),price_mae=np.mean(score_p['ae']),
                                      wind_mean_rmse=np.sqrt(np.mean(score_w['squared'])),price_mean_rmse=np.sqrt(np.mean(score_p['squared'])),
                                      wind_cover90=np.mean(score_w['cover90']),price_cover90=np.mean(score_p['cover90']),
                                      trajectory_energy=energy_trajectory(wind,price,wt,realw,realp),
                                      scen_wind_price_rho=float(wt @ row_spearman(wind,price)),
                                      observed_wind_price_rho=float(spearmanr(realw,realp).statistic),
                                      wind_low10_brier=float(np.mean((np.sum(wt[:,None]*(wind<10),axis=0)-(realw<10))**2)),
                                      price_high100_brier=float(np.mean((np.sum(wt[:,None]*(price>100),axis=0)-(realp>100))**2)),
                                      high_price_low_wind_brier=float(np.mean((np.sum(wt[:,None]*((wind<40)&(price>75)),axis=0)-((realw<40)&(realp>75)))**2)),
                                      actual_wind_low10_rate=float(np.mean(realw<10)),actual_price_high100_rate=float(np.mean(realp>100))))
    d=pd.DataFrame(daily)
    h=pd.DataFrame(hourly)
    if d.empty:raise RuntimeError('No evaluation rows')
    d.to_csv(out/'daily_metrics.csv',index=False)
    h.to_csv(out/'hourly_metrics.csv',index=False)
    pd.DataFrame([{k:v for k,v in obj.items() if k!='meta'} for obj in issuance]).to_csv(out/'issuance_audit.csv',index=False)
    (out/'issuance_metadata.json').write_text(json.dumps(issuance,indent=2,default=str))
    metrics=['wind_crps','price_crps','wind_mae','price_mae','wind_mean_rmse','price_mean_rmse','wind_cover90','price_cover90','trajectory_energy','wind_low10_brier','price_high100_brier','high_price_low_wind_brier','scen_wind_price_rho','observed_wind_price_rho']
    results=d.groupby(['ensemble','week','method'],as_index=False)[metrics].mean()
    results.to_csv(out/'weekly_metrics.csv',index=False)
    total=d.groupby(['ensemble','method'],as_index=False)[metrics].mean()
    total.to_csv(out/'overall_metrics.csv',index=False)
    lead=h.groupby(['ensemble','group','method'],as_index=False)[['wind_crps','price_crps','wind_ae','price_ae','wind_cover90','price_cover90','wind_width90','price_width90']].mean()
    lead.to_csv(out/'leadtime_metrics.csv',index=False)
    record=dict(issued_days=int(d[['week','day']].drop_duplicates().shape[0]),weeks=args.weeks,forecast_horizon_h=84,scored_hours=([24] if args.weeks==['feb_2026_24h'] else [84]),raw=args.raw,reduced=args.keep,
                seed=args.seed,checkpoint_hashes=recorded,
                protocol='D-1 09:30 America/Chicago, 84h from next local midnight, each day evaluated; overlapping horizons',
                source='frozen shipped continuous_CVAE_v3 calibrated vs shipped causal historical empirical copula',
                data_scope='Winter 2026 historical holdout; actual 2026 context supplied only as-of issue time' if all('2026' in k for k in args.weeks) else '2025 post-calibration historical comparison; not 2026 untouched holdouts',
                caveats=['Copula pool assembled from entire pre-2026 development CSV, but filters eligible analogues to blocks fully finished by issuance.',
                         'CVAE weights learned through 2024, validation Jan-Feb 2025, calibration Mar-Jun 2025; the copula assimilates past 2025 analogues and recent DA prices.',
                         'Hourly score records from successive daily 84-hour forecasts overlap; not independent hourly samples.',
                         'Only forecast quality measured; the PTES optimisation is not rerun.',
                         'February 2026 has complete 84h outcomes for February 9-12 issuance only; 24h outcomes for February 9-14.'])
    (out/'comparison_method.json').write_text(json.dumps(record,indent=2))
    print('\nOVERALL\n',total.round(3).to_string(index=False),flush=True)
    print('\nWEEKLY\n',results.round(3).to_string(index=False),flush=True)
    print('\nOutput',out)

if __name__=='__main__':main()
