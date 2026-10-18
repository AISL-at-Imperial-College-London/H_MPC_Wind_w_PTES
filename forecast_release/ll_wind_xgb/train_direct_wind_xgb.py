"""Direct 48-increment Horse Creek XGBoost wind predictor (paper-style).

Train: genuine 5-minute wind 2025-12-05 06:00 to 2026-01-20 23:55 UTC.
Validation: 2026-01-21 to Jan-28 UTC, allowed predeployment development ONLY.
Never reads any 2026 deployment data. Strictly uses timestamps < issue.

Outputs direct_xgb_wind.json, metrics.json, by_horizon.csv.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

N=48
START=pd.Timestamp('2025-12-05T06:00:00Z')
TRAIN_END=pd.Timestamp('2026-01-21T00:00:00Z')
VALID_END=pd.Timestamp('2026-01-29T00:00:00Z')


def make_windows(series,train_stride=3,valid_stride=1):
    """At issue index i: wind[i-49:i] (past) -> wind[i:i+48] (future)."""
    from numpy.lib.stride_tricks import sliding_window_view
    df=series.copy()
    df.index=pd.to_datetime(df.index,utc=True)
    df=df.sort_index().loc[START:VALID_END-pd.Timedelta(minutes=5)].asfreq('5min')
    w=df.to_numpy(float)
    raw=sliding_window_view(w,N*2+1)  # [i-49 ... i+47]
    origins=np.arange(N+1,len(w)-N+1)
    # scipy/xgb training on missing values: fail rather than interpolate
    x=np.diff(raw[:,:N+1],axis=1)  # 48 past increments (i-49 -> i-1)
    y=np.diff(raw[:,N:],axis=1)    # 48 future increments (i-1 -> i+47)
    anchor=raw[:,N]
    times=df.index[origins]
    valid=np.isfinite(raw).all(axis=1) & (raw.min(axis=1)>=0) & (raw.max(axis=1)<=230)
    training=(times+pd.Timedelta(minutes=5*N)<=TRAIN_END)&valid&((origins%train_stride)==0)
    testing=(times>=TRAIN_END)&(times+pd.Timedelta(minutes=5*N)<=VALID_END)&valid&((origins%valid_stride)==0)
    return (times[training],x[training],y[training],anchor[training]), (times[testing],x[testing],y[testing],anchor[testing])


def forecast(model,X,anchor):
    dy=model.predict(X)
    if dy.ndim!=2 or dy.shape[1]!=N:raise ValueError('Expected 48 output increments')
    return np.clip(anchor[:,None]+np.cumsum(dy,axis=1),0,230)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--trees',type=int,default=100)
    p.add_argument('--stride',type=int,default=3)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    dat=pd.read_csv(a.data,parse_dates=['timestamp_utc']).set_index('timestamp_utc')
    tr,val=make_windows(dat['HorseCreek_MW'],train_stride=a.stride)
    print('Train origins',len(tr[0]),'Validation origins',len(val[0]),flush=True)
    model=XGBRegressor(n_estimators=a.trees,max_depth=5,learning_rate=.05,
             objective='reg:squarederror',tree_method='hist',multi_strategy='multi_output_tree',
             reg_lambda=10.0,subsample=.9,colsample_bytree=1.,random_state=47,n_jobs=4)
    model.fit(tr[1],tr[2],verbose=False)
    model.save_model(out/'direct_xgb_wind.json')
    pred=forecast(model,val[1],val[3]);real=np.clip(val[3][:,None]+np.cumsum(val[2],axis=1),0,230)
    baseline=np.repeat(val[3][:,None],N,axis=1)
    mae_xgb=np.abs(pred-real).mean(axis=0);mae_p=np.abs(baseline-real).mean(axis=0)
    ramp_mask=np.abs(real[:,11]-val[3])>=30
    ramp_xgb=float(np.abs(pred[ramp_mask,:12]-real[ramp_mask,:12]).mean()) if ramp_mask.any() else None
    ramp_p=float(np.abs(baseline[ramp_mask,:12]-real[ramp_mask,:12]).mean()) if ramp_mask.any() else None
    metrics={'train_origins':len(tr[0]),'validation_origins':len(val[0]),
        'train_end_exclusive':str(TRAIN_END),'validation_end_exclusive':str(VALID_END),
        'xgb_mae_4h_MW':float(mae_xgb.mean()),'persistence_mae_4h_MW':float(mae_p.mean()),
        'xgb_mae_first_hour_MW':float(mae_xgb[:12].mean()),'persistence_mae_first_hour_MW':float(mae_p[:12].mean()),
        'realized_60min_ramp_ge30_origins':int(ramp_mask.sum()),'xgb_mae_first_hour_on_ramp_MW':ramp_xgb,
        'persistence_mae_first_hour_on_ramp_MW':ramp_p,
        'architecture':'100-tree native XGBoost multi_output_tree; 48 lagged 5min increments -> 48 future increments',
        'training_data':'genuine 5-minute Horse Creek measurement; no interpolation',
        'validation_data':'2026-01-21..28 predesignated development, no deployment'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    pd.DataFrame({'lead_minutes':np.arange(1,N+1)*5,'xgb_mae_MW':mae_xgb,'persistence_mae_MW':mae_p}).to_csv(out/'by_horizon.csv',index=False)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
