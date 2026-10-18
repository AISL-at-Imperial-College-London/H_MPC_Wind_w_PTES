"""2026 deployment data adapter and strictly causal forecast history.

All controller equations remain those in the frozen December packages.
The only predeclared exceptions are initial forecast/risk bootstrap, missing
measurement carry-forward and *unscored* trailing local-day completion.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SEASONS={
 'winter':('2026-02-01','2026-02-09T00:00:00Z','2026-02-16T00:00:00Z'),
 'spring':('2026-05-03','2026-05-11T00:00:00Z','2026-05-18T00:00:00Z'),
 'summer':('2026-07-05','2026-07-13T00:00:00Z','2026-07-20T00:00:00Z'),
}
ACTIVE=None

def _read(season,basename,value):
    path=ROOT/'holdout_inputs'/season/basename
    df=pd.read_csv(path)
    df['timestamp_utc']=pd.to_datetime(df['timestamp_utc'],utc=True,format='mixed')
    df=df.set_index('timestamp_utc').sort_index()
    if df.index.has_duplicates:raise ValueError('Duplicate historical timestamp in '+str(path))
    return pd.to_numeric(df[value],errors='coerce'),df

def load_season(season):
    if season not in SEASONS:raise ValueError(f'Unknown season {season!r}')
    local,score0,score1=SEASONS[season]
    score0=pd.Timestamp(score0);score1=pd.Timestamp(score1)
    wind,wdf=_read(season,'wind_5min.csv','HorseCreek_MW')
    da,ddf=_read(season,'DA_hourly.csv','DA_price_usd_mwh')
    rt,rdf=_read(season,'RT15.csv','RT15_price_usd_mwh')
    # Require 5/15/60min complete grid; do not silently drop gaps.
    wgrid=pd.date_range(wind.index[0],score1-pd.Timedelta(minutes=5),freq='5min',tz='UTC')
    dgrid=pd.date_range(da.index[0],score1-pd.Timedelta(hours=1),freq='h',tz='UTC')
    rgrid=pd.date_range(rt.index[0],score1-pd.Timedelta(minutes=15),freq='15min',tz='UTC')
    wind=wind.reindex(wgrid);da=da.reindex(dgrid);rt=rt.reindex(rgrid)
    if da.isna().any():raise ValueError('Missing DA price (cannot safely impute settled hours)')
    # Causal measurement repair: missing five-minute observation is replaced by
    # the last previously observed value, never a backward interpolation.
    missing_wind=wind[wind.isna()].index
    if len(missing_wind) and wind.notna().idxmax() >= missing_wind[0]:
        raise ValueError('Missing wind at start, cannot causal forward-fill')
    wind=wind.ffill()
    missing_rt=rt[rt.isna()].index
    if len(missing_rt) and rt.notna().idxmax() >= missing_rt[0]:
        raise ValueError('Missing RT at start, cannot causal forward-fill')
    rt=rt.ffill()
    for name,s in [('wind',wind),('DA',da),('RT',rt)]:
        if s.isna().any():raise ValueError(f'NaN remains in {name} after declared repair')
    if not wind.between(0,230).all():raise ValueError('Wind outside 0..230 MW')
    # The scored windows end at UTC midnight, six (winter) or five (CDT)
    # hours BEFORE the final local market day ends. The original unchanged
    # market-day runner needs an entire local day and four hours of forecast DA.
    # Append *pre-context constants*, available before the FIRST warm-up bid.
    # These are NEVER considered observed/settled holdout measurements and
    # NEVER contribute to scored metrics. All later actions are excluded.
    # Using a fixed pre-context constant prevents lookahead into scored data.
    fixed_w=float(wind.iloc[0]);fixed_da=float(da.iloc[0]);fixed_rt=float(rt.iloc[0])
    tail_end=score1+pd.Timedelta(hours=16)
    wind=pd.concat([wind,pd.Series(fixed_w,index=pd.date_range(score1,tail_end-pd.Timedelta(minutes=5),freq='5min',tz='UTC'))])
    da=pd.concat([da,pd.Series(fixed_da,index=pd.date_range(score1,tail_end-pd.Timedelta(hours=1),freq='h',tz='UTC'))])
    rt=pd.concat([rt,pd.Series(fixed_rt,index=pd.date_range(score1,tail_end-pd.Timedelta(minutes=15),freq='15min',tz='UTC'))])
    hourly=pd.DataFrame({'DA_price_usd_mwh':da})
    return {'season':season,'first_market_day':local,'score_start':score0,'score_end':score1,
            'wind':wind,'rt':rt,'hourly':hourly,'source_end':score1,
            'wind_repairs':[t.isoformat() for t in missing_wind],
            'rt_repairs':[t.isoformat() for t in missing_rt],
            'tail_policy':'unscored post-UTC-window dummy constants fixed from first context hour',
            'tail_source_values':dict(wind_mw=fixed_w,DA_USD_MWh=fixed_da,RT_USD_MWh=fixed_rt)}

def hourly_runtime_history(active,cutoff):
    t=pd.Timestamp(cutoff).tz_convert('UTC')
    wind=active['wind'].loc[active['wind'].index+pd.Timedelta(minutes=5)<=t]
    # The current hourly wind value must NOT use an incomplete hour.
    by_hour=wind.groupby(wind.index.floor('h')).agg(['mean','count'])
    wind_hour=by_hour.loc[by_hour['count']==12,'mean']
    available_da=active['hourly']['DA_price_usd_mwh']
    hourly=pd.DataFrame({'HorseCreek_MW':wind_hour,'DA_price_usd_mwh':available_da.reindex(wind_hour.index)})
    hourly=hourly.loc[hourly.index+pd.Timedelta(hours=1)<=t].dropna()
    if hourly.empty:raise ValueError('No causal 2026 hourly observations')
    last_completed=t.floor('h')-pd.Timedelta(hours=1)
    needed=pd.date_range(end=last_completed,periods=96,freq='h',tz='UTC')
    # Only initial warm-up bid is short of 96 hours; explicitly carry earliest
    # CONTEXT observations backwards. This is a deterministic bootstrap, not
    # unseen historical data and not used as a training sample.
    absent=needed[~needed.isin(hourly.index)]
    if len(absent):
        if max(absent)>=hourly.index.min() or len(absent)>36:
            raise ValueError(f'Insufficient non-contiguous 96h as-of history at {t}; gaps={len(absent)}')
        oldest=hourly.iloc[0]
        pads=pd.DataFrame({col:float(oldest[col]) for col in hourly.columns},index=absent)
        hourly=pd.concat([pads,hourly]).sort_index()
    if hourly.reindex(needed).isna().any().any():
        raise ValueError(f'Unresolved T-CVAE runtime-history gaps at {t}')
    return hourly.loc[hourly.index<t].copy(),len(absent)

def install_adapters(active):
    """Called before each replay; no rewriting of trained forecasting models."""
    global ACTIVE
    ACTIVE=active
    import run_multiday as pt
    import wind_only.runner as wo
    import ptes_mpc.hl_market as ph
    import wind_only.historical_risk as wh
    pt.load_raw=lambda:(active['hourly'],active['rt'],active['wind'])
    wo.load_raw=lambda:(active['hourly'],active['rt'],active['wind'])
    # Temporarily allow ONLY the declared deployment market-day replay. The
    # source December entrypoints remain protected against accidental unblinding.
    def allowed_holdout_days(first,days):
        d=pd.Timestamp(first)
        if d.tzinfo is not None or int(days)!=15:
            raise ValueError('Holdout replay requires exactly fifteen market days')
        allowed={pd.Timestamp(value[0]) for value in SEASONS.values()}
        if d not in allowed:
            raise ValueError('Market-day start is not one of the three protected deployment starts')
        return [(d+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(15)]
    pt.validate_days=allowed_holdout_days
    wo.validate_days=allowed_holdout_days

    # Both source runners import same named forecasting module from same package.
    import sys
    forecast_dir=ROOT/'forecast_release'/'source_releases'/'horse_creek_forecasting_release'
    if str(forecast_dir) not in sys.path:sys.path.insert(0,str(forecast_dir))
    import horse_creek_forecasts as fc
    if not hasattr(fc.JointForecastProvider,'_pre_holdout_generate'):
        fc.JointForecastProvider._pre_holdout_generate=fc.JointForecastProvider.generate
    def causal_generate(self,delivery_day,*,runtime_history=None,n_raw=500,n_keep=50,seed=3714):
        from horse_creek_copula import delivery_times
        cutoff,_=delivery_times(delivery_day)
        runtime,extra=hourly_runtime_history(ACTIVE,cutoff)
        result=self._pre_holdout_generate(delivery_day,runtime_history=runtime,
                                         n_raw=n_raw,n_keep=n_keep,seed=seed)
        result.metadata['holdout_initial_context_bootstrap_hours']=int(extra)
        return result
    fc.JointForecastProvider.generate=causal_generate
    # Original 28d spread estimator is preserved when >=5d real 2026 history.
    # For the first 1-2 warm-up bids only, use fixed PRE-2026 development
    # financial spread profile rather than future 2026 settlement observations.
    original=ph.estimate_spread_from_past
    if getattr(original,'_holdout_wrapper',False): original=original._original
    def historic_spread(*,rt15,da_hourly,decision_utc,delivery_times_utc,window_days=28):
        try:
            return original(rt15=rt15,da_hourly=da_hourly,decision_utc=decision_utc,
                            delivery_times_utc=delivery_times_utc,window_days=window_days)
        except ValueError as exc:
            if 'insufficient historical' not in str(exc) and 'not enough local-hour' not in str(exc):raise
            dev=ROOT/'forecast_release'/'source_releases'/'horse_creek_forecasting_release'/'data'
            oldda=pd.read_csv(dev/'development_master_hourly_2018_2021_2024_2025.csv',parse_dates=['timestamp_utc']).set_index('timestamp_utc')['DA_price_usd_mwh'].sort_index()
            oldrt=pd.read_csv(dev/'rt15_all_development_2018_2021_2025.csv',parse_dates=['timestamp_utc']).set_index('timestamp_utc')['RT15_price_usd_mwh'].sort_index()
            # Earlier data only, never from any 2026 protected week.
            ref_issue=pd.Timestamp('2025-12-31T23:45:00Z')
            return original(rt15=oldrt,da_hourly=oldda,decision_utc=ref_issue,
                            delivery_times_utc=delivery_times_utc,window_days=window_days)
    historic_spread._holdout_wrapper=True
    historic_spread._original=original
    pt.estimate_spread_from_past=historic_spread
    wh.estimate_spread_from_past=historic_spread
    return pt,wo
