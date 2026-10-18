#!/usr/bin/env python3
"""Horse Creek RT15 *development* benchmark. Uses 2025 Jan-Jun training,
Jul-Aug validation, Sep-Dec development confirmation. No 2026 data.

Model forecasts RT15-DA; DA target is required to be for the same local delivery
calendar day as the issue, to avoid making assumptions about next-day DAM
result-release times. All historical RT inputs are delayed after settlement
interval end by a user-specified *assumption*, not verified release timestamps.

Run: python forecast_rt15_spread.py --data-dir /path/to/development_data
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

NLEADS=16
CUTOFFS={'train_end':'2025-07-01T00:00:00Z','validation_end':'2025-09-01T00:00:00Z','test_end':'2026-01-01T00:00:00Z'}


def base_tables(data_dir):
    r=pd.read_csv(data_dir/'rt15_all_development_2018_2021_2025.csv', parse_dates=['timestamp_utc'])
    r=r.loc[(r.timestamp_utc>=pd.Timestamp('2025-01-01',tz='UTC'))&(r.timestamp_utc<pd.Timestamp('2026-01-01',tz='UTC'))]
    rt=r.set_index('timestamp_utc').sort_index().RT15_price_usd_mwh.asfreq('15min')
    if len(rt)!=35040 or rt.isna().any():raise RuntimeError(f'2025 RT missing: length={len(rt)}, nans={rt.isna().sum()}')
    h=pd.read_csv(data_dir/'development_master_hourly_2018_2021_2024_2025.csv',parse_dates=['timestamp_utc'])
    h=h.set_index('timestamp_utc').sort_index()
    da=h.DA_price_usd_mwh.loc['2025-01-01':'2025-12-31'].asfreq('h')
    wind=h.HorseCreek_MW.loc['2025-01-01':'2025-12-31'].asfreq('h')
    return rt, da, wind


def make_table(rt,da,wind,lag_min=60):
    """Align one feature row per issue time and target settlement interval.
    Last allowed RT interval start = issue-(15+publication_lag) minutes.
    Last allowed hourly wind start = issue.floor(hour)-1h (complete).
    2025 UTC year only; all target prices and DA must exist.
    """
    if lag_min<0 or lag_min%15:raise ValueError('Publication lag must be nonnegative multiple of 15 min')
    ts=rt.index
    issue_idx=np.arange(len(rt))
    source_j=(lag_min+15)//15
    # Historical RT values, shifted by 1+delay: at decision t, last interval start t-75min for delay=60.
    past_rt={f'rt_lag_{k}':rt.shift(source_j+k).to_numpy() for k in [0,1,2,4,8,16,32]}
    rt_lag=past_rt['rt_lag_0']
    past_rt['rt_recent_mean_1h']=rt.shift(source_j).rolling(4,min_periods=4).mean().to_numpy()
    past_rt['rt_recent_mean_4h']=rt.shift(source_j).rolling(16,min_periods=16).mean().to_numpy()
    past_rt['rt_recent_std_4h']=rt.shift(source_j).rolling(16,min_periods=16).std().to_numpy()
    # DA price for most recent published-eligible RT interval; historical DA itself is already cleared.
    da_at_eligible=da.reindex((ts-pd.Timedelta(minutes=15+lag_min)).floor('h')).to_numpy()
    rtspread=rt_lag-da_at_eligible
    past_rt['spread_lag_0']=rtspread
    for k in [1,2,4,8,16]:
        src=past_rt[f'rt_lag_{k}']
        da_src=da.reindex((ts-pd.Timedelta(minutes=15+lag_min+15*k)).floor('h')).to_numpy()
        past_rt[f'spread_lag_{k}']=src-da_src
    # Historical wind hourly blocks must be fully completed at decision time. Missing wind stays NaN for this diagnostic.
    w0=wind.reindex(ts.floor('h')-pd.Timedelta(hours=1)).to_numpy()
    w1=wind.reindex(ts.floor('h')-pd.Timedelta(hours=2)).to_numpy()
    w4=wind.reindex(ts.floor('h')-pd.Timedelta(hours=5)).to_numpy()
    base=pd.DataFrame({'issue_utc':ts,'rt_last':rt_lag,'wind_last_complete':w0,
                       'wind_d1h':w0-w1,'wind_d4h':w0-w4,
                       'rt_minus_da_last':rtspread})
    for k,v in past_rt.items():base[k]=v
    base['recent_abs_spread']=np.abs(base.rt_minus_da_last)
    # Feature of current DA clearing, available for current settlement day only.
    daissue=da.reindex(ts.floor('h')).to_numpy()
    base['da_current']=daissue
    local_t=ts.tz_convert('America/Chicago')
    data=[]
    for j in range(NLEADS):
        target=ts+pd.Timedelta(minutes=15*j)
        loc=target.tz_convert('America/Chicago')
        # Strict safe subset: DA for current *local* calendar day already cleared the preceding day.
        same_day=(loc.date==local_t.date)
        y=rt.reindex(target).to_numpy()
        da_tar=da.reindex(target.floor('h')).to_numpy()
        z=base.copy()
        z['target_utc']=target
        z['target_rt']=y
        z['da_target']=da_tar
        z['rt_da_spread']=y-da_tar
        z['DA_delta_target_vs_current']=da_tar-daissue
        z['lead_min']=15*(j+1)
        z['target_hour_local']=loc.hour
        z['target_minute_local']=loc.minute
        z['target_hour_sin']=np.sin(2*np.pi*(loc.hour+loc.minute/60)/24)
        z['target_hour_cos']=np.cos(2*np.pi*(loc.hour+loc.minute/60)/24)
        z['weekday_sin']=np.sin(2*np.pi*loc.dayofweek/7)
        z['weekday_cos']=np.cos(2*np.pi*loc.dayofweek/7)
        z['month_sin']=np.sin(2*np.pi*(loc.month-1)/12)
        z['month_cos']=np.cos(2*np.pi*(loc.month-1)/12)
        z=z.loc[same_day]
        data.append(z)
    out=pd.concat(data,ignore_index=True)
    mask=out[['rt_da_spread','DA_delta_target_vs_current','rt_lag_32','rt_recent_std_4h',
              'wind_last_complete','wind_d1h','wind_d4h']].notna().all(axis=1)
    out=out.loc[mask].copy()
    out['day_local']=out.issue_utc.dt.tz_convert('America/Chicago').dt.strftime('%Y-%m-%d')
    out['split']=np.where(out.target_utc<pd.Timestamp(CUTOFFS['train_end']),'train',
                          np.where(out.target_utc<pd.Timestamp(CUTOFFS['validation_end']),'validation',
                                   np.where(out.target_utc<pd.Timestamp(CUTOFFS['test_end']),'test','excluded')))
    return out[out.split!='excluded'].reset_index(drop=True)


COLS=['da_target','da_current','DA_delta_target_vs_current','lead_min',
       'target_hour_sin','target_hour_cos','target_minute_local','weekday_sin',
       'weekday_cos','month_sin','month_cos','wind_last_complete','wind_d1h','wind_d4h',
       'rt_lag_0','rt_lag_1','rt_lag_2','rt_lag_4','rt_lag_8','rt_lag_16','rt_lag_32',
       'rt_recent_mean_1h','rt_recent_mean_4h','rt_recent_std_4h','spread_lag_0',
       'spread_lag_1','spread_lag_2','spread_lag_4','spread_lag_8','spread_lag_16',
       'recent_abs_spread']
COLS_DA_ONLY=[x for x in COLS if x not in ('rt_lag_0','rt_lag_1','rt_lag_2','rt_lag_4','rt_lag_8','rt_lag_16','rt_lag_32','rt_recent_mean_1h','rt_recent_mean_4h','rt_recent_std_4h','spread_lag_0','spread_lag_1','spread_lag_2','spread_lag_4','spread_lag_8','spread_lag_16','recent_abs_spread')]


def score(df):
    preds=['DA_only','RT_persistence','rt_spread_shrink','XGB_DA_wind','XGB_RT_spread']
    metrics=[]
    for label,z in [('all',df),('15-60min',df[df.lead_min<=60]),('75-240min',df[df.lead_min>60]),
                    ('large_spread_realized_ge50',df[(df.target_rt-df.da_target).abs()>=50]),
                    ('large_spread_realized_ge100',df[(df.target_rt-df.da_target).abs()>=100])]:
        for method in preds:
            err=z[method]-z.target_rt
            metrics.append({'subset':label,'model':method,'n':len(z),'MAE':float(err.abs().mean()),
                            'bias':float(err.mean()),'RMSE':float(np.sqrt((err**2).mean())),
                            'MAE_spread_from_DA':float(err.abs().mean())})
    return pd.DataFrame(metrics)


def bootstrap(df,n=400,seed=2026):
    rng=np.random.default_rng(seed)
    grouped=df.groupby('day_local',sort=True)
    days=list(grouped.groups)
    if not len(days):return {}
    out={}
    for method in ('XGB_DA_wind','XGB_RT_spread','rt_spread_shrink'):
        daily=grouped.apply(lambda x:pd.Series({'n':len(x),'total_delta':((x[method]-x.target_rt).abs()-(x.DA_only-x.target_rt).abs()).sum()}),include_groups=False)
        vals=[]
        for _ in range(n):
            ix=rng.choice(len(daily),size=len(daily),replace=True)
            v=daily.iloc[ix]
            vals.append(float(v.total_delta.sum()/v.n.sum()))
        out[method]={'mae_minus_DA_only':float(daily.total_delta.sum()/daily.n.sum()),
                     '95pct_CI':np.quantile(vals,[0.025,0.975]).round(4).tolist(),
                     'n_days':len(daily)}
    return out


def fit_and_test(frame,outdir):
    train=frame[frame.split=='train'].copy();val=frame[frame.split=='validation'].copy();test=frame[frame.split=='test'].copy()
    print('Windows', {name:len(z) for name,z in [('train',train),('val',val),('test',test)]},flush=True)
    print('Target-period separation',train.target_utc.max(),val.target_utc.min(),val.target_utc.max(),test.target_utc.min(),flush=True)
    y=train.rt_da_spread.clip(-150,150).to_numpy()
    mod={}
    for name,cols in [('XGB_DA_wind',COLS_DA_ONLY),('XGB_RT_spread',COLS)]:
        model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.055,
                          subsample=.85,colsample_bytree=.9,reg_lambda=12,
                          objective='reg:squarederror',tree_method='hist',n_jobs=4,random_state=101)
        print('Training',name,flush=True)
        model.fit(train[cols].to_numpy(np.float32),y,verbose=False)
        mod[name]=model
        model.save_model(outdir/f'{name}_development_2025.json')
    # Fit shrinkage factor ONLY on train, using zero-intercept ridge with light penalty.
    x=train.rt_minus_da_last.to_numpy(float);yraw=train.rt_da_spread.to_numpy(float)
    alpha=float(np.clip((x@yraw)/(x@x+1.e-8),0,1))
    print('Train-only spread persistence shrink coefficient alpha',alpha,flush=True)
    outall={}
    for label,d in [('validation',val),('test',test)]:
        d=d.copy()
        d['DA_only']=d.da_target
        d['RT_persistence']=d.rt_lag_0
        d['rt_spread_shrink']=d.da_target+alpha*d.rt_minus_da_last
        for name,cols in [('XGB_DA_wind',COLS_DA_ONLY),('XGB_RT_spread',COLS)]:
            d[name]=d.da_target+mod[name].predict(d[cols].to_numpy(np.float32))
        summary=score(d)
        summary.insert(0,'period',label)
        summary.to_csv(outdir/f'{label}_metrics.csv',index=False)
        pd.DataFrame({'issue_utc':d.issue_utc,'target_utc':d.target_utc,'lead_min':d.lead_min,
                      'observed_rt':d.target_rt,'DA_only':d.DA_only,'RT_persistence':d.RT_persistence,
                      'rt_spread_shrink':d.rt_spread_shrink,'XGB_DA_wind':d.XGB_DA_wind,
                      'XGB_RT_spread':d.XGB_RT_spread}).to_csv(outdir/f'{label}_predictions.csv',index=False)
        bs=bootstrap(d)
        (outdir/f'{label}_day_block_uncertainty.json').write_text(json.dumps(bs,indent=2))
        print('\n',label,' overall and large spreads:\n',summary[summary.subset.isin(['all','large_spread_realized_ge50'])].to_string(index=False,float_format=lambda x:f'{x:.3f}'),flush=True)
        print('Day-block bootstrap:',bs,flush=True)
        outall[label]=d
    settings={'training_window':'2025 Jan-Jun UTC','validation_window':'2025 Jul-Aug UTC',
              'development_test_window':'2025 Sep-Dec UTC (previously analysed as part of 2025, NOT untouched)',
              'RT_publication_lag_minutes_assumed':lag_min,'price_target':'RT15 minus cleared DA',
              'target_clip_train_+/-USD_per_MWh':150,'RT_lag_latest_start':'issue - 15min - assumed_lag',
              'DA_safe_delivery_time':'Only forecast RT target intervals on same local America/Chicago calendar day as issue',
              'features_DA_only':COLS_DA_ONLY,'features_RThistory':COLS,'alpha_train':alpha,
              '2026_deployment_used':False,'2024_RT_available':False}
    (outdir/'audit_and_settings.json').write_text(json.dumps(settings,indent=2))
    return outall,settings


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',type=Path,default=(Path(__file__).resolve().parent/'data'))
    ap.add_argument('--lag',type=int,default=60)
    ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    global lag_min
    lag_min=args.lag
    rt,da,wind=base_tables(args.data_dir)
    df=make_table(rt,da,wind,lag_min=args.lag)
    d,settings=fit_and_test(df,args.output)
    # Learning curve / forecast lead comparison
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label,data in d.items():
        f=data.groupby('lead_min').apply(lambda x:pd.Series({k:(x[k]-x.target_rt).abs().mean() for k in ['DA_only','RT_persistence','rt_spread_shrink','XGB_DA_wind','XGB_RT_spread']}),include_groups=False)
        f.to_csv(args.output/f'{label}_mae_by_horizon.csv')
        fig,ax=plt.subplots(figsize=(9,5))
        for m in f.columns:ax.plot(f.index,f[m],marker='.',label=m)
        ax.set(xlabel='RT settlement lead time (minutes)',ylabel='MAE ($/MWh)',title=f'ERCOT HORSECRK_RN 2025 {label}: Strict timing; lag {lag_min}min')
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(args.output/f'{label}_error_by_horizon.png',dpi=160);plt.close(fig)
    print('All artifacts in',args.output,flush=True)

if __name__=='__main__':main()
