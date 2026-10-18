"""Causal 15-minute settlement-price XGBoost provider + 5-minute MPC cache.

Frozen candidate: zero-delay 2025 XGBoost (model and training/evaluation code
bundled separately). The latest *completed* quarter-hour RT15 SPP is assumed
observed at the next quarter-hour origin; this is a modelling timing assumption,
not a claim that ERCOT publishes final settlement prices instantaneously.

Reforecast only at 15-minute boundaries. At +5/+10min, keep same price
forecast for each 15-minute settlement bucket. DA-only tail if needed when
mapping a 16-interval forecast onto 48 five-minute operating steps.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

RT_LAGS=(0,1,2,4,8,16,32)
SPREAD_LAGS=(0,1,2,4,8,16)
COLS=['da_target','da_current','DA_delta_target_vs_current','lead_min',
       'target_hour_sin','target_hour_cos','target_minute_local','weekday_sin',
       'weekday_cos','month_sin','month_cos','wind_last_complete','wind_d1h','wind_d4h',
       'rt_lag_0','rt_lag_1','rt_lag_2','rt_lag_4','rt_lag_8','rt_lag_16','rt_lag_32',
       'rt_recent_mean_1h','rt_recent_mean_4h','rt_recent_std_4h','spread_lag_0',
       'spread_lag_1','spread_lag_2','spread_lag_4','spread_lag_8','spread_lag_16',
       'recent_abs_spread']


def _series(obj,possible):
    if isinstance(obj,pd.DataFrame):
        found=next((k for k in possible if k in obj.columns),None)
        if found is None:raise ValueError(f'Missing column among {possible}')
        s=obj[found].copy()
    elif isinstance(obj,pd.Series):s=obj.copy()
    else:raise TypeError('Expected timestamp-indexed Series or DataFrame')
    s.index=pd.to_datetime(s.index,utc=True)
    if s.index.duplicated().any():raise ValueError('Duplicate time index')
    return s.sort_index().astype(float)


class RT15XGBoostProvider:
    def __init__(self,model_path=None):
        self.model_path=Path(model_path) if model_path else Path(__file__).parent/'xgb_zero_delay_2025_trainJanJun.json'
        self.model=XGBRegressor()
        self.model.load_model(self.model_path)

    def _inputs(self,issue_utc,observed_rt15,observed_wind_hourly,cleared_da_hourly):
        t=pd.Timestamp(issue_utc)
        if t.tz is None:raise ValueError('Issue time must be tz aware')
        t=t.tz_convert('UTC')
        if t.minute%15 or t.second or t.microsecond:raise ValueError('RT15 XGB issues only at 15-minute boundaries')
        rt=_series(observed_rt15,['RT15_price_usd_mwh','rt'])
        wind=_series(observed_wind_hourly,['HorseCreek_MW','wind'])
        da=_series(cleared_da_hourly,['DA_price_usd_mwh','da'])
        if len(rt)==0 or rt.index.max()>=t:raise ValueError('RT history contains current/future interval')
        if len(wind) and (wind.index.max()+pd.Timedelta(hours=1)>t):
            raise ValueError('Wind hourly history contains incomplete hour')
        if len(da)==0:raise ValueError('No cleared DA')
        # Cleared DA is permitted for future periods ONLY with caller's verified
        # clearing-time attestation; the archive does not prove publish time.
        rtvals={k:float(rt.get(t-pd.Timedelta(minutes=15*(k+1)),np.nan)) for k in RT_LAGS}
        prices=np.array([float(rt.get(t-pd.Timedelta(minutes=15*k),np.nan)) for k in range(1,17)])
        if not np.isfinite(prices).all():raise ValueError('Need complete last 16 RT15 observations')
        wbase=t.floor('h')-pd.Timedelta(hours=1)
        w0=float(wind.get(wbase,np.nan));w1=float(wind.get(wbase-pd.Timedelta(hours=1),np.nan))
        w4=float(wind.get(wbase-pd.Timedelta(hours=4),np.nan))
        da_current=float(da.get(t.floor('h'),np.nan))
        base={'da_current':da_current,'wind_last_complete':w0,'wind_d1h':w0-w1,'wind_d4h':w0-w4,
              'rt_recent_mean_1h':float(prices[:4].mean()),
              'rt_recent_mean_4h':float(prices.mean()),
              'rt_recent_std_4h':float(prices.std(ddof=1))}
        for k,x in rtvals.items():
            base[f'rt_lag_{k}']=x
            if k in SPREAD_LAGS:
                historical_hour=(t-pd.Timedelta(minutes=15*(k+1))).floor('h')
                base[f'spread_lag_{k}']=x-float(da.get(historical_hour,np.nan))
        base['recent_abs_spread']=abs(base['spread_lag_0'])
        targets=pd.date_range(t,periods=16,freq='15min',tz='UTC')
        local_issue=t.tz_convert('America/Chicago')
        X=[];same_day=[]
        for j,tar in enumerate(targets):
            loc=tar.tz_convert('America/Chicago')
            row=base.copy()
            da_target=float(da.get(tar.floor('h'),np.nan))
            row.update(da_target=da_target,DA_delta_target_vs_current=da_target-da_current,
                       lead_min=15*(j+1),target_hour_sin=np.sin(2*np.pi*(loc.hour+loc.minute/60)/24),
                       target_hour_cos=np.cos(2*np.pi*(loc.hour+loc.minute/60)/24),
                       target_minute_local=loc.minute,
                       weekday_sin=np.sin(2*np.pi*loc.dayofweek/7),
                       weekday_cos=np.cos(2*np.pi*loc.dayofweek/7),
                       month_sin=np.sin(2*np.pi*(loc.month-1)/12),
                       month_cos=np.cos(2*np.pi*(loc.month-1)/12))
            X.append(row);same_day.append(loc.date()==local_issue.date())
        X=pd.DataFrame(X,columns=COLS,dtype=float)
        if not np.isfinite(X.to_numpy()).all():raise ValueError('Missing historical RT/wind or cleared DA observations')
        return t,targets,X,np.asarray(same_day)

    def forecast(self,issue_utc,observed_rt15,observed_wind_hourly,cleared_da_hourly,*,da_cleared_at_utc,method='xgb'):
        t,targets,X,same=self._inputs(issue_utc,observed_rt15,observed_wind_hourly,cleared_da_hourly)
        ct=pd.Timestamp(da_cleared_at_utc)
        if ct.tz is None or ct.tz_convert('UTC')>t:raise ValueError('DA clearing attestation must predate issue')
        if method=='da_only':p=X['da_target'].to_numpy(dtype=float, copy=True)
        elif method=='xgb':
            # Pandas Copy-on-Write can expose a read-only view. The RT spread
            # is added in place below, so an explicitly writable copy is required.
            p=X['da_target'].to_numpy(dtype=float, copy=True)
            if same.any():p[same]+=self.model.predict(X.loc[same,COLS].to_numpy(np.float32))
            # Crossing local midnight: a deliberately documented DA-only fallback.
        elif method=='rt_persistence':p=np.full(16,X.loc[0,'rt_lag_0'])
        else:raise ValueError('method xgb/da_only/rt_persistence')
        return {'issued_utc':t,'starts_utc':targets,'RT15_price_USD_MWh':p,
                'model':method,'local_midnight_fallback_count':int((~same).sum()),
                'RT_last_interval_start_utc':(t-pd.Timedelta(minutes=15)).isoformat(),
                'assumes_RT15_immediately_available_after_interval_end':True,
                'DA_asof_utc':ct.isoformat()}


class RT15QuarterHourCache:
    """Retain a 15-min issue forecast through two intermediate 5-min solves."""
    def __init__(self):self._last=None

    def refresh(self,pred):
        self._last=pred

    def at_5min_issue(self,issue_utc,cleared_da_hourly):
        if self._last is None:raise ValueError('Call refresh at an RT 15-minute boundary first')
        t=pd.Timestamp(issue_utc)
        if t.tz is None:raise ValueError('Need timezone-aware issue')
        t=t.tz_convert('UTC')
        if t.second or t.microsecond or t.minute%5:raise ValueError('Issue must be 5-min boundary')
        refreshed=self._last['issued_utc']
        if t<refreshed or t>=refreshed+pd.Timedelta(minutes=15):
            raise ValueError('Stale RT price forecast; refresh at quarter hour')
        da=_series(cleared_da_hourly,['DA_price_usd_mwh','da'])
        targets=pd.date_range(t,periods=48,freq='5min',tz='UTC')
        prices=pd.Series(self._last['RT15_price_USD_MWh'],index=self._last['starts_utc'])
        vals=[];tail=0
        for q in targets:
            s=q.floor('15min')
            if s in prices: vals.append(float(prices.loc[s]))
            else:
                v=float(da.get(q.floor('h'),np.nan))
                if not np.isfinite(v):raise ValueError('Missing cleared DA tail for 5-min horizon')
                vals.append(v);tail+=1
        return {'issued_utc':t,'forecast_last_refreshed_utc':refreshed,
                'starts_utc':targets,'RT_expected_USD_MWh':np.array(vals),
                'DA_fallback_tail_steps':tail,'RT_refresh_due':t.minute%15==0}
