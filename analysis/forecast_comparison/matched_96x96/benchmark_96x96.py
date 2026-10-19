#!/usr/bin/env python3
"""Causal apples-to-apples comparison: frozen continuous CVAE versus 96/96 empirical analogue copula.

Neither CVAE architecture nor deployed checkpoints are modified. Both forecast the 96 hours
immediately after their last 96 completed wind & price measurements. The analogue model
uses feature-based similarity in BOTH 96-hour histories, 96-hour causal price recalibration,
and a fixed reference library ending 2025-06-30. This is a NEW experimental copula baseline.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT_DEFAULT=Path(__file__).parent/'horse-creek-ptes-mpc'
SEASONS={'February':['2026-02-09','2026-02-10','2026-02-11','2026-02-12'],
         'May':['2026-05-11','2026-05-12','2026-05-13','2026-05-14'],
         'July':['2026-07-13','2026-07-14','2026-07-15','2026-07-16']}
TRAIN_END=pd.Timestamp('2025-06-30 23:00:00',tz='UTC')
FUTURE_H=96
HISTORY_H=96


def features(raw):
    """96h wind and price history -> fixed, scale-compatible seasonal analogue features.

    All four 24h blocks are used, plus last-24h volatility and last-12h mean.
    Input shape (n,96,2), chronological oldest to newest. Price uses asinh to
    keep rare spikes from overwhelming state conditioning.
    """
    a=np.asarray(raw, dtype=float)
    if a.ndim!=3 or a.shape[1:]!=(96,2):raise ValueError('Bad feature input')
    w=a[:,:,0]/230.
    p=np.arcsinh(a[:,:,1]/50.)
    wm=w.reshape(-1,4,24).mean(2)
    pm=p.reshape(-1,4,24).mean(2)
    ws=np.std(w[:,-24:],axis=1)[:,None]
    ps=np.std(p[:,-24:],axis=1)[:,None]
    wl=w[:,-12:].mean(1)[:,None]
    pl=p[:,-12:].mean(1)[:,None]
    return np.column_stack([wm,pm,ws,ps,wl,pl])

class EmpiricalAnalogue96:
    def __init__(self, history, training_end=TRAIN_END):
        self.dev=history.loc[history.index<=training_end].copy()
        self.training_end=training_end
        idx=self.dev.index
        vals=self.dev[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(float)
        begin=[];month=[];old=[];future=[]
        t0=time.perf_counter()
        for j in range(96,len(self.dev)-96+1):
            ts=idx[j]
            if ts.tz_convert('America/Chicago').hour !=9:continue
            subset=idx[j-96:j+96]
            if len(subset)!=192 or not np.all(np.diff(subset.asi8)==3600*10**9):continue
            v=vals[j-96:j+96]
            if not np.isfinite(v).all():continue
            begin.append(ts); month.append(ts.tz_convert('America/Chicago').month)
            old.append(v[:96]);future.append(v[96:])
        self.starts=pd.DatetimeIndex(begin)
        self.months=np.asarray(month)
        self.past=np.stack(old)
        self.future=np.stack(future)
        self.feat=features(self.past)
        print(f'Analogue library: {len(self.starts)} complete 96+96 historical trajectories, ending {self.starts.max()} ({time.perf_counter()-t0:.1f}s)',flush=True)
    def generate(self, past, first_unobserved, n=500,seed=3714):
        past=np.asarray(past,dtype=float)
        if past.shape!=(96,2) or not np.isfinite(past).all():raise ValueError('Require valid 96h wind+price history')
        ts=pd.Timestamp(first_unobserved)
        month=ts.tz_convert('America/Chicago').month
        m=np.abs(self.months-month)
        ids=np.flatnonzero(np.minimum(m,12-m)<=2)
        if len(ids)<40:raise ValueError(f'Insufficient seasonal analogues: {len(ids)}')
        target=features(past[None])[0]
        # Fixed scales chosen BEFORE testing on 2026. Channels share comparable
        # influence; recent daily blocks receive more weight than older ones.
        std=np.maximum(np.std(self.feat[ids],axis=0),[.10]*4+[.20]*4+[.07,.13,.10,.20])
        weight=np.asarray([.40,.55,.75,1.0]*2+[.55,.55,1.,1.])
        z=(self.feat[ids]-target)/std
        d=np.sqrt(np.sum(weight*z*z,axis=1)/np.sum(weight))
        # Stable Gaussian kernel with nonzero uniform mixture like the original.
        probability=.15+np.exp(-.5*d*d)
        probability/=probability.sum()
        rng=np.random.default_rng(seed)
        picked=rng.choice(ids,n,replace=True,p=probability)
        wind=self.future[picked,:,0].copy()
        price=self.future[picked,:,1].copy()
        # Quantile-map price marginal to same 96 COMPLETED hourly inputs as CVAE.
        ref=np.sort(self.future[ids,:,1].ravel())
        recent=np.sort(past[:,1])
        ranks=np.searchsorted(ref,price,side='right')/(len(ref)+1)
        grid=(np.arange(96)+.5)/96
        price=np.interp(np.clip(ranks,grid[0],grid[-1]),grid,recent)
        meta={'eligible_analogue_count':int(len(ids)),'latest_analogue_start_utc':str(self.starts[ids].max()),
              'reference_training_end_utc':self.training_end.isoformat(),
              'matched_recent_wind_history_h':96,'matched_recent_price_history_h':96,
              'price_quantile_target_observations':96,'training_excludes_all_holdout_data':True,
              'features':'4 daily wind means, 4 daily transformed-price means, last-day channel std, last-12h channel means',
              'feature_kernel':'Gaussian on prior-96h feature distance plus 0.15 uniform analogue floor',
              'sampling_seed':int(seed)}
        return wind,price,meta



def _weighted_quantile(paths, weights, level):
    order=np.argsort(paths,axis=0)
    pp=np.take_along_axis(paths,order,axis=0)
    ww=np.take_along_axis(np.broadcast_to(weights[:,None],paths.shape),order,axis=0)
    idx=np.minimum(np.argmax(np.cumsum(ww,axis=0)>=level,axis=0),len(paths)-1)
    return pp[idx,np.arange(paths.shape[1])]

def _weighted_crps(paths,weights,truth):
    err=np.sum(weights[:,None]*np.abs(paths-truth[None,:]),axis=0)
    order=np.argsort(paths,axis=0)
    p=np.take_along_axis(paths,order,axis=0)
    w=np.take_along_axis(np.broadcast_to(weights[:,None],paths.shape),order,axis=0)
    left_weight=np.cumsum(w,axis=0)-w
    left_sum=np.cumsum(w*p,axis=0)-w*p
    correction=np.sum(w*(p*left_weight-left_sum),axis=0)
    return err-correction

def score_group(paths,weights,truth):
    mu=np.sum(weights[:,None]*paths,axis=0)
    lo=_weighted_quantile(paths,weights,.05)
    hi=_weighted_quantile(paths,weights,.95)
    return {'crps':_weighted_crps(paths,weights,truth),
            'cover90':((truth>=lo)&(truth<=hi)).astype(float),
            'ae':np.abs(mu-truth)}

def energy_trajectory(wind,price,weights,truth_w,truth_p):
    # Same dimensionless 96h joint-vector transformation as earlier comparison.
    series=np.column_stack([wind/230.,np.arcsinh(price/50.)])
    target=np.r_[truth_w/230.,np.arcsinh(truth_p/50.)]
    return float(weights@cdist(series,target[None,:]).ravel()
                 -.5*weights@cdist(series,series)@weights)

def weighted_scores(wind, price,weights, actual):
    # Score implementation is bundled below; no dependence on earlier comparison scripts.
    sw=score_group(wind,weights,actual[:,0])
    sp=score_group(price,weights,actual[:,1])
    return dict(wind_crps=float(sw['crps'].mean()),price_crps=float(sp['crps'].mean()),
                wind_cover90=float(sw['cover90'].mean()),price_cover90=float(sp['cover90'].mean()),
                wind_mae=float(sw['ae'].mean()),price_mae=float(sp['ae'].mean()),
                energy_score=energy_trajectory(wind,price,weights,actual[:,0],actual[:,1]))


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',type=Path,default=ROOT_DEFAULT)
    a.add_argument('--out',type=Path,default=Path(__file__).parent/'results_96x96')
    a.add_argument('--seeds',nargs='+',type=int,default=[3714,13714,2718])
    a.add_argument('--n-raw',type=int,default=500)
    a.add_argument('--n-keep',type=int,default=50)
    a.add_argument('--max-origins',type=int,default=4,help='limit forecasts per week for a quicker smoke test')
    args=a.parse_args()
    repo=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    mod=repo/'forecast_release/source_releases/horse_creek_forecasting_release'
    sys.path.insert(0,str(mod))
    from horse_creek_forecasts import DATA,MODEL,CALIBRATION,prepare_hourly,delivery_times
    from continuous_cvae_provider import ContinuousCalibratedCVAE
    from original_backward_reduction import backward_reduce_original
    development=prepare_hourly(pd.read_csv(DATA))
    empirical=EmpiricalAnalogue96(development)
    cvae=ContinuousCalibratedCVAE(MODEL,CALIBRATION)
    data={}
    for key,dname in [('February','winter'),('May','spring'),('July','summer')]:
        frame=pd.read_csv(repo/f'holdout_inputs/{dname}/HL_realized_hourly_wind_DA.csv')
        frame=frame.rename(columns={'HorseCreek_MW_hourly':'HorseCreek_MW'})
        data[key]=prepare_hourly(frame)
    rows=[]; audit=[]
    for seed in args.seeds:
        for key,days in SEASONS.items():
            source=data[key]
            for day in days[:args.max_origins]:
                cutoff,deliver=delivery_times(day)
                latest=cutoff.floor('h')-pd.Timedelta(hours=1)
                past_idx=pd.date_range(end=latest,periods=96,freq='h',tz='UTC')
                past=source.reindex(past_idx)
                first=latest+pd.Timedelta(hours=1)
                target_idx=pd.date_range(first,periods=96,freq='h',tz='UTC')
                truth=source.reindex(target_idx)
                if not (past.notna().all().all() and truth.notna().all().all()):
                    raise ValueError(f'{day} missing 96h input/forecast target')
                known=past[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(float)
                actual=truth[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(float)
                for method in ['copula_96x96','deployed_cvae_96x96_eval']:
                    t=time.perf_counter()
                    if method=='copula_96x96':
                        wind,price,detail=empirical.generate(known,first,n=args.n_raw,seed=seed)
                    else:
                        wind120,price120=cvae.generate_raw(known,first,n=args.n_raw,seed=seed)
                        wind,price=wind120[:,:96].copy(),price120[:,:96].copy()
                        detail={'original_checkpoint_forecast_h':120, 'evaluated_forecast_h':96}
                    keep,probs=backward_reduce_original(wind,price,args.n_keep)
                    for label,ww,pp,pr in [(f'raw_{args.n_raw}',wind,price,np.full(args.n_raw,1/args.n_raw)),
                                            (f'reduced_{args.n_keep}',wind[keep],price[keep],probs)]:
                        metrics=weighted_scores(ww,pp,pr,actual)
                        rows.append(dict(seed=seed,week=key,day=day,method=method,ensemble=label,**metrics))
                    audit.append(dict(seed=seed,week=key,day=day,method=method,
                                      cutoff_utc=cutoff.isoformat(),first_forecast_hour_utc=first.isoformat(),
                                      last_forecast_hour_utc=target_idx[-1].isoformat(),
                                      last_known_hour_start_utc=latest.isoformat(),
                                      num_input_hours=len(past),num_target_hours=len(truth),
                                      duration_sec=round(time.perf_counter()-t,2),**detail))
                    print(f'{seed} {key} {day} {method} {time.perf_counter()-t:.1f}s',flush=True)
    result=pd.DataFrame(rows);result.to_csv(out/'daily_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'issuance_audit.csv',index=False)
    metrics=['wind_crps','price_crps','wind_cover90','price_cover90','wind_mae','price_mae','energy_score']
    weekly=result.groupby(['seed','week','ensemble','method'],as_index=False)[metrics].mean()
    pooled=result.groupby(['seed','ensemble','method'],as_index=False)[metrics].mean()
    weekly.to_csv(out/'weekly_metrics.csv',index=False)
    pooled.to_csv(out/'pooled_metrics.csv',index=False)
    multi=pooled.groupby(['ensemble','method'],as_index=False)[metrics].mean()
    multi.to_csv(out/'multi_seed_metrics.csv',index=False)
    manifest=dict(description=__doc__, seeds=args.seeds, training_end_utc=TRAIN_END.isoformat(),
                  cvAE_checkpoint_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                  calibration_sha256=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
                  pool_data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                  history_hours=96,scoring_hours=96,raw=args.n_raw,kept=args.n_keep,
                  selected_forecast_dates=SEASONS,origins_per_week=args.max_origins,
                  scenario_reduction='same shipped backward_reduce_original, but over the 96h evaluation window',
                  caution='This is NOT the original copula nor the exact MPC 84h scenario reduction or PTES closed-loop evaluation.')
    (out/'methodology.json').write_text(json.dumps(manifest,indent=2))
    print('FINAL\n',multi.to_string(index=False),flush=True)
    print('Wrote',out,flush=True)
if __name__=='__main__':main()
