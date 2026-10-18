"""Three historical development/reused-context weeks on the unchanged V3 MPC.

The extra-week data and shorter causal bootstrap follow the original
Horse_Creek_3_Extra_Weeks package. Models/control come from HC_MPC_V3.
This module only adapts input histories, eligible replay dates and the
price-spread history fallback; it does NOT change any solver formulation.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import holdout_runtime as hr

ROOT=Path(__file__).resolve().parent
SEASONS = {
    'dec_2025': ('2025-12-06','2025-12-08T00:00:00Z','2025-12-15T00:00:00Z'),
    'spring_dev': ('2026-05-01','2026-05-04T00:00:00Z','2026-05-11T00:00:00Z'),
    'summer_dev': ('2026-07-03','2026-07-06T00:00:00Z','2026-07-13T00:00:00Z'),
}
# Preserve the original named protected holdouts and add these supplementary dates.
hr.SEASONS.update(SEASONS)


def expected_market_days(season):
    first, _, end = SEASONS[season]
    last_local = pd.Timestamp(end).tz_convert('America/Chicago').date()
    return (last_local - pd.Timestamp(first).date()).days + 1


def run_metadata():
    return {
        'dec_2025': dict(period='8-14 Dec 2025 UTC', market_days=9,
            warmup_hours=42, initial_history_policy='earliest as-of hourly pair padding',
            data_status='development (December 2025 may overlap wind XGB training); NOT an independent holdout'),
        'spring_dev': dict(period='4-10 May 2026 UTC', market_days=10,
            warmup_hours=67, initial_history_policy='earliest as-of hourly pair padding',
            data_status='previous holdout context/warm-up re-used as scored historical replay; NOT an independent holdout'),
        'summer_dev': dict(period='6-12 July 2026 UTC', market_days=10,
            warmup_hours=67, initial_history_policy='earliest as-of hourly pair padding',
            data_status='previous holdout context/warm-up re-used as scored historical replay; NOT an independent holdout'),
    }


def load_season(season):
    if season not in SEASONS: raise ValueError(f'Invalid supplementary season {season}')
    return hr.load_season(season)


def history_available_at_bid(active, cutoff):
    """Exactly 96 completed historical hours; pad missing PRIOR hours only.

    A later scored/warmup observation is never used to back-fill. The first
    observed joint (wind, DA) hour is duplicated only into unavailable prior
    historical slots, matching the extra-weeks package convention.
    """
    t = pd.Timestamp(cutoff).tz_convert('UTC')
    w = active['wind'].loc[active['wind'].index + pd.Timedelta(minutes=5) <= t]
    by_hour = w.groupby(w.index.floor('h')).agg(['mean','count'])
    wind_hour = by_hour.loc[by_hour['count'] == 12,'mean']
    prices = active['hourly']['DA_price_usd_mwh']
    hist = pd.DataFrame({'HorseCreek_MW':wind_hour,
                         'DA_price_usd_mwh':prices.reindex(wind_hour.index)})
    hist = hist.loc[hist.index + pd.Timedelta(hours=1) <= t].dropna()
    if hist.empty: raise ValueError(f'No causal historical wind-price pair at {t}')
    needed = pd.date_range(end=t.floor('h')-pd.Timedelta(hours=1),
                           periods=96, freq='h', tz='UTC')
    absent=needed[~needed.isin(hist.index)]
    if len(absent):
        if max(absent) >= hist.index.min() or len(absent)>95:
            raise ValueError(f'Noncausal/noncontiguous extra-week history at {t}: {len(absent)} h')
        first=hist.iloc[0]
        hist=pd.concat([pd.DataFrame({c:float(first[c]) for c in hist.columns}, index=absent), hist]).sort_index()
    hist=hist.reindex(needed)
    if hist.isna().any().any(): raise ValueError(f'Unresolved historical context at {t}')
    return hist.copy(),len(absent)


def install_adapters(active):
    """Install original V3 causal forecast wrapper, adjust short extra episode.

    The historical spread fallback is explicitly applied to the imported name
    used by run_multiday AND the name used by wind_only.runner. This avoids
    accidentally using the V3 default 2025-12-31 reference for December 2025.
    """
    season=active['season']
    if season not in SEASONS:raise ValueError(f'Not an extra season: {season}')
    pt,wo=hr.install_adapters(active)
    hr.hourly_runtime_history=history_available_at_bid

    expected={pd.Timestamp(v[0]):expected_market_days(k) for k,v in SEASONS.items()}
    def allowed_days(first,days):
        d=pd.Timestamp(first)
        if d.tzinfo is not None or d not in expected or int(days)!=expected[d]:
            raise ValueError(f'Unexpected supplementary replay dates: {first} x {days}')
        dates=[(d+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(int(days))]
        for date in dates:
            a=pd.Timestamp(date).tz_localize('America/Chicago')
            b=(pd.Timestamp(date)+pd.Timedelta(days=1)).tz_localize('America/Chicago')
            if b-a!=pd.Timedelta(days=1):raise ValueError('DST transition in replay is not supported')
        return dates
    pt.validate_days=allowed_days
    wo.validate_days=allowed_days

    import ptes_mpc.hl_market as ph
    import wind_only.historical_risk as wh
    base=ph.estimate_spread_from_past
    if getattr(base,'_holdout_wrapper',False):base=base._original
    # Load earlier reference prices lazily, only when insufficient as-of data.
    def causal_spread_fallback(*,rt15,da_hourly,decision_utc,delivery_times_utc,window_days=28):
        try:
            return base(rt15=rt15,da_hourly=da_hourly,
                        decision_utc=decision_utc,delivery_times_utc=delivery_times_utc,
                        window_days=window_days)
        except ValueError as exc:
            if ('insufficient historical' not in str(exc) and
                'not enough local-hour' not in str(exc) and
                'no eligible completed historical' not in str(exc)):
                raise
            data=ROOT/'forecast_release'/'source_releases'/'horse_creek_forecasting_release'/'data'
            oldda=pd.read_csv(data/'development_master_hourly_2018_2021_2024_2025.csv',
                    parse_dates=['timestamp_utc']).set_index('timestamp_utc')['DA_price_usd_mwh'].sort_index()
            oldrt=pd.read_csv(data/'rt15_all_development_2018_2021_2025.csv',
                    parse_dates=['timestamp_utc']).set_index('timestamp_utc')['RT15_price_usd_mwh'].sort_index()
            ref=pd.Timestamp('2025-11-30T23:45:00Z' if season=='dec_2025'
                             else '2025-12-31T23:45:00Z')
            result=base(rt15=oldrt.loc[:ref],da_hourly=oldda.loc[:ref],
                        decision_utc=ref,delivery_times_utc=delivery_times_utc,
                        window_days=window_days)
            return result
    causal_spread_fallback._holdout_wrapper=True
    causal_spread_fallback._original=base
    ph.estimate_spread_from_past=causal_spread_fallback
    wh.estimate_spread_from_past=causal_spread_fallback
    pt.estimate_spread_from_past=causal_spread_fallback  # cached import must change
    return pt,wo
