"""Strict-as-of 5-minute XGBoost and persistence wind providers.

Assume timestamps label 5-minute observations; at decision t only timestamp <t
is available. Forecast values correspond to starts t,t+5min,...,t+235min.
"""
from pathlib import Path
import numpy as np
import pandas as pd

N=48
CAP_MW=230.0


def _wind(observations):
    if isinstance(observations,pd.DataFrame):
        if 'HorseCreek_MW' not in observations:raise ValueError('Expected HorseCreek_MW column')
        s=observations['HorseCreek_MW']
    elif isinstance(observations,pd.Series):s=observations
    else:raise TypeError('Pass timestamp-indexed pandas Series or DataFrame')
    s=s.copy()
    s.index=pd.to_datetime(s.index,utc=True)
    if s.index.duplicated().any():raise ValueError('Duplicate wind timestamp')
    return s.sort_index()


class Wind5MinuteProvider:
    def __init__(self,model_path=None):
        self.path=Path(model_path) if model_path else Path(__file__).parent/'direct_xgb_wind.json'
        self._model=None

    def predict(self,issue_utc,observed,method='xgb'):
        t=pd.Timestamp(issue_utc)
        if t.tz is None:raise ValueError('Issue time must be timezone-aware')
        t=t.tz_convert('UTC')
        if t.second or t.microsecond or (t.minute%5):raise ValueError('Wind forecast requires 5-minute origin')
        s=_wind(observed)
        if len(s) and s.index.max()>=t:raise ValueError('Wind history includes observation from issue/future')
        past=pd.date_range(end=t-pd.Timedelta(minutes=5),periods=49,freq='5min',tz='UTC')
        v=s.reindex(past).to_numpy(float)
        if not np.isfinite(v).all():raise ValueError('Missing genuine 5-minute measurements in 49-step lookback')
        if ((v<0)|(v>CAP_MW)).any():raise ValueError('Invalid wind power')
        if method=='persistence':out=np.full(N,v[-1])
        elif method=='xgb':
            if self._model is None:
                from xgboost import XGBRegressor
                self._model=XGBRegressor()
                self._model.load_model(self.path)
            inc=np.diff(v)[None,:].astype(np.float32)
            future_inc=self._model.predict(inc)[0]
            if len(future_inc)!=N:raise RuntimeError('Malformed 48-step model output')
            out=np.clip(v[-1]+np.cumsum(future_inc),0,CAP_MW)
        else:raise ValueError('method: xgb or persistence')
        return {'issued_utc':t, 'starts_utc':pd.date_range(t,periods=N,freq='5min',tz='UTC'),
                'wind_MW':out.astype(float),'method':method,'last_realized_wind_MW':float(v[-1])}
