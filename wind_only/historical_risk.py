"""Historical DA/RT price spread proxy, copied from submitted study's HL strategy.
It is a forward bidding risk/price estimator, NOT an actual imbalance tariff.
"""
import pandas as pd
import numpy as np

def estimate_spread_from_past(*,rt15,da_hourly,decision_utc,delivery_times_utc,
                               window_days=28):
    """Return as-of historic mean RT-DA spread and half IQR per local hour.

    Half-IQR is a predeclared proxy for imbalance exposure, NOT a settlement
    rule or deducted from realised cashflows. Never read prices >= decision.
    """
    cutoff=pd.Timestamp(decision_utc).tz_convert('UTC')
    a=pd.Series(rt15,dtype=float).sort_index();d=pd.Series(da_hourly,dtype=float).sort_index()
    a.index=pd.to_datetime(a.index,utc=True);d.index=pd.to_datetime(d.index,utc=True)
    a=a[(a.index+pd.Timedelta(minutes=15)<=cutoff)&(a.index>=cutoff-pd.Timedelta(days=window_days))]
    if a.empty:raise ValueError('no eligible completed historical RT data')
    dh=d.reindex(a.index.floor('h')).to_numpy(float)
    spread=a.to_numpy(float)-dh
    hours=a.index.tz_convert('America/Chicago').hour
    filt=np.isfinite(spread)
    if filt.sum()<4*24*5:raise ValueError('insufficient historical RT-DA spread history')
    means={};risks={}
    for h in range(24):
        x=spread[filt&(hours==h)]
        if len(x)<8:raise ValueError(f'not enough local-hour historic RT spread for {h}')
        means[h]=float(np.mean(x))
        risks[h]=float(max(0.01,(np.quantile(x,.75)-np.quantile(x,.25))/2))
    target_hours=pd.DatetimeIndex(delivery_times_utc).tz_convert('America/Chicago').hour
    return np.asarray([means[h] for h in target_hours]),np.asarray([risks[h] for h in target_hours])

