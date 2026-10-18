"""Adapters to the frozen 10-Oct-2026 forecasting release (not bundled here).

Root path must point to extracted horse_creek_forecasting_MPC_handover_2026-10-10.
Calls do not access actual future values, and build completed hourly wind from 5min
observations at issue time. RT immediate publication assumption is preserved.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def hourly_from_completed_5min(wind_5min, issue_utc):
    s=pd.Series(wind_5min).sort_index()
    t=pd.Timestamp(issue_utc).tz_convert('UTC')
    if len(s) and s.index.max()>=t:raise ValueError('future 5-minute wind observations')
    if len(s)==0:return pd.Series(dtype=float,index=pd.DatetimeIndex([],tz='UTC'))
    first=s.index.min().ceil('h')
    last=(t-pd.Timedelta(hours=1)).floor('h')
    result={}
    for hour in pd.date_range(first,last,freq='h',tz='UTC'):
        idx=pd.date_range(hour,periods=12,freq='5min',tz='UTC')
        vals=s.reindex(idx).to_numpy(float)
        if np.isfinite(vals).all():result[hour]=float(vals.mean())
    return pd.Series(result,dtype=float)

class FrozenProviders:
    def __init__(self,forecast_release_root,*,da_cleared_at_for,
                 wind_method='xgb',rt_method='xgb'):
        root=Path(forecast_release_root).resolve()
        for sub in ['ll_wind_xgb','ll_rt_xgb']:
            if not (root/sub).exists():raise FileNotFoundError(f'forecast package missing {sub}')
            sys.path.insert(0,str(root/sub))
        from wind_runtime import Wind5MinuteProvider
        from rt_xgb_runtime import RT15XGBoostProvider
        self._wind=Wind5MinuteProvider(root/'ll_wind_xgb'/'direct_xgb_wind.json')
        self._rt=RT15XGBoostProvider(root/'ll_rt_xgb'/'xgb_zero_delay_2025_trainJanJun.json')
        self._da_cleared_at_for=da_cleared_at_for
        self.wind_method=wind_method
        self.rt_method=rt_method

    def wind(self,issue_utc,measured_wind):
        return self._wind.predict(issue_utc,measured_wind,self.wind_method)['wind_MW']

    def rt(self,issue_utc,observed_rt15,cleared_da,measured_wind):
        complete=hourly_from_completed_5min(measured_wind,issue_utc)
        return self._rt.forecast(issue_utc,observed_rt15,complete,cleared_da,
             da_cleared_at_utc=self._da_cleared_at_for(issue_utc),
             method=self.rt_method)['RT15_price_USD_MWh']
