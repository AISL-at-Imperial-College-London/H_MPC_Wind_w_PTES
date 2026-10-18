#!/usr/bin/env python3
"""Evaluate conditional sensitivity of a newly trained thesis-style T-CVAE.

Read-only: uses pre-2026 development data. Does not retrain or run MPC.
Generates plots, case-wise forecast diagnostics, and a same-calendar comparison.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.dates import DateFormatter, DayLocator
from train_thesis_tcvae import (ThesisTCVAE, load_hourly_source, inverse_transform,
                                HISTORY, FUTURE, RATING, TZ)


def _scaler_encode(x, scale):
    x=np.asarray(x,float).copy()
    if scale['price_transform']=='asinh': x[...,1]=np.arcsinh(x[...,1]/50.0)
    return ((x-np.asarray(scale['mean']))/np.asarray(scale['std'])).astype(np.float32)


def load_model(path,device):
    # Load checkpoints ONLY produced locally by this training package.
    cp=torch.load(path,map_location=device,weights_only=False)
    model=ThesisTCVAE(hidden=cp['hidden'],latent_dim=cp['latent_dim']).to(device)
    model.load_state_dict(cp['model']);model.eval()
    return model,cp['scaling']


def predict(model,scale,source,date,seed,n,device):
    """D-1 09:30 local forecast: only 96 completed hourly rows are conditioning input."""
    local_midnight=pd.Timestamp(str(date)+' 00:00').tz_localize(TZ)
    start=local_midnight.tz_convert('UTC')
    cutoff=(local_midnight-pd.Timedelta(days=1)+pd.Timedelta(hours=9,minutes=30)).tz_convert('UTC')
    history_end=cutoff.floor('h')-pd.Timedelta(hours=1)
    hist_idx=pd.date_range(end=history_end,periods=96,freq='h',tz='UTC')
    future_idx=pd.date_range(start=start,periods=96,freq='h',tz='UTC')
    history=source.reindex(hist_idx)[['HorseCreek_MW','DA_price_usd_mwh']]
    actual=source.reindex(future_idx)[['HorseCreek_MW','DA_price_usd_mwh']]
    if history.isna().any().any() or actual.isna().any().any():
        raise ValueError(f'{date} lacks complete historical or future 96-hour data')
    if hist_idx[-1]+pd.Timedelta(hours=1)>cutoff:
        raise RuntimeError('Forecast issuance lookahead detected')
    x=_scaler_encode(history.to_numpy(float)[None],scale)
    x=torch.tensor(x,device=device)
    d=torch.tensor([local_midnight.dayofweek],device=device)
    h=torch.tensor([local_midnight.hour],device=device)
    gen=inverse_transform(model.sample(x,d,h,n=n,seed=seed),scale)
    return dict(date=str(date),history=history,actual=actual,hist_idx=hist_idx,future_idx=future_idx,
                cutoff=cutoff,start=start,scenarios=gen)


def crps_ensemble(x,y):
    # x: S x T, y: T. CRPS = E|X-y| - 0.5 E|X-X'|.
    n=x.shape[0]
    term=np.abs(x-y[None]).mean(0)
    # Efficient O(S log S) sorted pairwise difference sum for every time step
    xs=np.sort(x,axis=0)
    coeff=(2*np.arange(n)-n+1).reshape(n,1)
    pair_half=(coeff*xs).sum(0)/(n*n)
    return float((term-pair_half).mean())


def evaluate_dates(model,scale,source,device,first=2025,last=2025,seed=1234,n=64,max_dates=48):
    """Chronologically spaced 2025 dates to check conditional responsiveness."""
    dates=pd.date_range('2025-07-05','2025-11-25',freq='3D')
    if len(dates)>max_dates:
        dates=dates[np.linspace(0,len(dates)-1,max_dates).round().astype(int)]
    rows=[]
    for date in dates:
        try:r=predict(model,scale,source,date.strftime('%Y-%m-%d'),seed,n,device)
        except ValueError:continue
        w=r['scenarios'][:,:24,0]
        p=r['scenarios'][:,:24,1]
        obs=r['actual'].to_numpy(float)[:24]
        past=r['history'].to_numpy(float)
        rows.append({
            'date':r['date'],
            'last24_wind_mw':float(past[-24:,0].mean()),
            'pred_first24_wind_mw':float(w.mean()),
            'actual_first24_wind_mw':float(obs[:,0].mean()),
            'wind_mae_first24_mw':float(np.abs(w.mean(0)-obs[:,0]).mean()),
            'wind_persistence_mae_first24_mw':float(np.abs(past[-1,0]-obs[:,0]).mean()),
            'wind_crps_first24_mw':crps_ensemble(w,obs[:,0]),
            'price_crps_first24_usd_mwh':crps_ensemble(p,obs[:,1]),
            'wind_90pct_coverage':float(np.mean((obs[:,0]>=np.quantile(w,.05,axis=0))&(obs[:,0]<=np.quantile(w,.95,axis=0)))),
            'price_90pct_coverage':float(np.mean((obs[:,1]>=np.quantile(p,.05,axis=0))&(obs[:,1]<=np.quantile(p,.95,axis=0)))),
        })
    if not rows:raise ValueError('No 2025 test origin has complete history and target')
    return pd.DataFrame(rows)


def weighted_plot(r,out):
    fig,axes=plt.subplots(2,1,figsize=(10.5,6.8),sharex=True)
    palette=[('#448a59','Day-ahead price (USD/MWh)'),('#357caf','Wind power (MW)')]
    # The history between issue and delivery is shown for continuity, but is not model input.
    df=r['history'];hist=r['hist_idx'];t=r['future_idx']
    available_hist=hist[-1]
    all_observed=r['source'].loc[(r['source'].index>=hist[0])&(r['source'].index<t[0])]
    for i,ax in enumerate(axes):
        color,ylabel=palette[i]
        plot_name=['DA_price_usd_mwh','HorseCreek_MW'][i]
        ax.axvspan(hist[0],r['cutoff'],color='#eeeeee',alpha=.9)
        ax.axvspan(r['cutoff'],t[0],color='#fafafa',alpha=.95)
        ax.plot(all_observed.index,all_observed[plot_name],color='black',lw=1.5,label='Observed')
        s=r['scenarios'][:,:,1 if i==0 else 0]
        q10,q25,q75,q90=np.quantile(s,[.1,.25,.75,.9],axis=0)
        for path in s[:min(50,len(s))]:ax.plot(t,path,color=color,alpha=.10,lw=.5)
        ax.fill_between(t,q10,q90,color=color,alpha=.13)
        ax.fill_between(t,q25,q75,color=color,alpha=.25)
        ax.plot(t,s.mean(0),ls='--',color=color,lw=1.7)
        ax.plot(t,r['actual'].iloc[:,1 if i==0 else 0],color='#db2b28',lw=1.6)
        ax.axvline(r['cutoff'],color='#db2b28',ls='--',lw=1.1)
        ax.axvline(r['start'],color='#444444',ls=':',lw=1.1)
        ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        if i==1:ax.set_ylim(bottom=0,top=240)
    legend=[Line2D([0],[0],color='black',lw=1.5,label='Historical observations'),
            Line2D([0],[0],color='#db2b28',ls='--',label='Forecast issue time'),
            Line2D([0],[0],color='#444',ls=':',label='First delivery hour'),
            Patch(facecolor='#448a59',alpha=.13,label='10th-90th percentile'),
            Patch(facecolor='#448a59',alpha=.25,label='25th-75th percentile'),
            Line2D([0],[0],color='#357caf',ls='--',label='Scenario mean'),
            Line2D([0],[0],color='#db2b28',label='Realization')]
    fig.legend(handles=legend,loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.5,.0))
    axes[-1].xaxis.set_major_locator(DayLocator(interval=1,tz=r['start'].tz))
    axes[-1].xaxis.set_major_formatter(DateFormatter('%d %b\n%H:%M',tz=r['start'].tz))
    axes[-1].set_xlabel('UTC time')
    fig.suptitle(f"T-CVAE scenario generation: {r['date']} (96h history / 96h future)",fontsize=11)
    fig.subplots_adjust(hspace=.3,bottom=.22,top=.93)
    fig.savefig(out/'scenario_example.png',dpi=160,bbox_inches='tight')
    fig.savefig(out/'scenario_example.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='Canonical development CSV or training ZIP')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='TCVAE_DIAGNOSTICS')
    p.add_argument('--day',default='2025-07-14')
    p.add_argument('--scenarios',type=int,default=64)
    p.add_argument('--dates',type=int,default=48)
    p.add_argument('--seed',type=int,default=3714)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    model,scaling=load_model(args.checkpoint,device)
    df,_=load_hourly_source(args.input)
    df=df.drop_duplicates('timestamp_utc').set_index('timestamp_utc').sort_index()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    r=predict(model,scaling,df,args.day,args.seed,args.scenarios,device)
    r['source']=df
    weighted_plot(r,out)
    daily=evaluate_dates(model,scaling,df,device,seed=args.seed,n=args.scenarios,max_dates=args.dates)
    daily.to_csv(out/'conditional_diagnostics_by_date.csv',index=False)
    def corr(a,b):
        if np.std(a)<1e-9 or np.std(b)<1e-9:return None
        return float(np.corrcoef(a,b)[0,1])
    summary={
        'n_dates':len(daily),
        'correlation_predicted_first24_vs_last24_wind':corr(daily.pred_first24_wind_mw,daily.last24_wind_mw),
        'correlation_predicted_first24_vs_actual_first24_wind':corr(daily.pred_first24_wind_mw,daily.actual_first24_wind_mw),
        'std_forecast_daily_mean_mw':float(daily.pred_first24_wind_mw.std()),
        'std_actual_daily_mean_mw':float(daily.actual_first24_wind_mw.std()),
        'mean_wind_mae_mw':float(daily.wind_mae_first24_mw.mean()),
        'mean_wind_persistence_mae_mw':float(daily.wind_persistence_mae_first24_mw.mean()),
        'mean_wind_crps_mw':float(daily.wind_crps_first24_mw.mean()),
        'mean_price_crps_usd_mwh':float(daily.price_crps_first24_usd_mwh.mean()),
        'mean_wind_90coverage':float(daily.wind_90pct_coverage.mean()),
        'mean_price_90coverage':float(daily.price_90pct_coverage.mean()),
        'qualification':'Diagnostic evaluation on 2025 development data; NOT the 2026 reserved MPC holdouts'
    }
    (out/'conditioning_summary.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots(figsize=(6,4))
    ax.scatter(daily.last24_wind_mw,daily.pred_first24_wind_mw,alpha=.8,label='Model predicted mean')
    ax.scatter(daily.last24_wind_mw,daily.actual_first24_wind_mw,alpha=.35,label='Realized daily mean')
    ax.set(xlabel='Last 24h observed wind mean (MW)',ylabel='Next-day 24h wind mean (MW)',
           title='Conditioning response: varying histories')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'conditioning_sensitivity.png',dpi=160);plt.close(fig)
    print('Saved conditioning diagnostics in',out)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
