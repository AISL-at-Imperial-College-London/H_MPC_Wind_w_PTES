"""Fit continuous CVAE-v3 trajectory residual bootstrap on PRE-TEST March-June 2025 alone.

Truth is used only after its entire 120-hour future has passed; calibration is
frozen before 1 July 2025, and NO 2026 data is fitted or inspected.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np,pandas as pd,torch
from continuous_cvae import *

def load_history_future(df,delivery_day):
    from datetime import timedelta
    from zoneinfo import ZoneInfo
    # Identical issue/target definitions to MPC horse_creek_copula.delivery_times.
    local=pd.Timestamp(delivery_day)
    start=local.tz_localize('America/Chicago')
    tomorrow=(local+pd.DateOffset(days=1)).tz_localize('America/Chicago')
    if (tomorrow.tz_convert('UTC')-start.tz_convert('UTC'))!=pd.Timedelta(hours=24):return None
    previous=(local-pd.DateOffset(days=1)).tz_localize('America/Chicago')
    cutoff=(previous+pd.Timedelta(hours=9,minutes=30)).tz_convert('UTC')
    start=start.tz_convert('UTC')
    histidx=pd.date_range(end=cutoff.floor('h')-pd.Timedelta(hours=1),periods=96,freq='h',tz='UTC')
    futidx=pd.date_range(histidx[-1]+pd.Timedelta(hours=1),periods=HORIZON,freq='h',tz='UTC')
    # Entire continuous path starts at first unobserved hour.
    if futidx[0] != histidx[-1]+pd.Timedelta(hours=1):return None
    raw=df.reindex(histidx)[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(float)
    future=df.reindex(futidx)[['HorseCreek_MW','DA_price_usd_mwh']].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(future).all():return None
    return raw,calendar_features([futidx[0]])[0],future

def fit(args):
    torch.set_num_threads(args.threads)
    df=pd.read_csv(args.input);df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    df=df.set_index('timestamp_utc').sort_index()
    model,cp=checkpoint_load(args.checkpoint);sc=cp['scaling']
    # Fitted only 2025-03-01 .. 2025-06-26 issues, and all 120h outcomes before 1 Jul.
    days=pd.date_range('2025-03-01','2025-06-24',freq='D')
    residuals=[];dates=[];means=[]
    for i,day in enumerate(days):
        observation=load_history_future(df,day.strftime('%Y-%m-%d'))
        if observation is None:continue
        h,cal,y=observation
        with torch.no_grad(): draw=make_raw(model,sc,h,cal,seed=8000+i,n=128)
        m=np.mean(draw,axis=0)
        residuals.append(transform(y,sc)-m)
        means.append(m)
        dates.append(str(day.date()))
    res=np.asarray(residuals,dtype=np.float32)
    if len(res)<50:raise RuntimeError('Insufficient pre-2025-Jul calibration days')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Frozen empirical 120h wind+price *paired* residual trajectories.
    np.savez_compressed(out/'continuous_calibration_residual_paths.npz',residuals=res,dates=np.asarray(dates))
    p={'n_calibration_origins':len(res),'first':dates[0],'last':dates[-1],
       'calibration_date_range':'2025-03-01 to 2025-06-26',
       'calibration_year':'2025 only','no_2026_data':True,
       'calibration_method':'joint historical 120h residual trajectories in train-normalized wind and asinh(price) space',
       'calibration_window_is_separate_from_training':True,
       'scenario_formula':'base CVAE mean + sampled empirical residual path + 0.15 * CVAE centred path',
       'cv_sample_scale':.15,'model_checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
       'technical_note':'Bootstrap provides coherent paired trajectories, but 90% future coverage is not guaranteed without stationarity.'}
    (out/'calibration_manifest.json').write_text(json.dumps(p,indent=2))
    print('Saved',len(res),'calibration paths; residual SD by variable',np.std(res,axis=(0,1)))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--checkpoint',required=True)
    a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=4)
    fit(a.parse_args())
