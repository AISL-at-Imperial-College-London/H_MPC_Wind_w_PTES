"""Wind-only historical development dates, market clocks, and input validation."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
FORECAST=ROOT/'forecast_release'
FORECAST_HL=FORECAST/'source_releases'/'horse_creek_forecasting_release'
DATA=FORECAST_HL/'data'
CHICAGO='America/Chicago'

def local_midnight(day):
    return pd.Timestamp(day).tz_localize(CHICAGO).tz_convert('UTC')

def validate_days(first, days):
    d=pd.Timestamp(first)
    if d.tz is not None or not (pd.Timestamp('2025-12-08')<=d<=pd.Timestamp('2025-12-23')):
        raise ValueError('Restricted to the development dates 8–23 Dec 2025; never open blind holdouts')
    if not 1<=days<=7 or d+pd.Timedelta(days=days-1)>pd.Timestamp('2025-12-23'):
        raise ValueError('Select 1–7 eligible consecutive development days')
    dates=[(d+pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    for day in dates:
        if local_midnight((pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%d'))-local_midnight(day)!=pd.Timedelta(hours=24):
            raise ValueError('DST transition not supported by 24h DA formulation')
    return dates

def load_raw():
    hourly=pd.read_csv(DATA/'development_master_hourly_2018_2021_2024_2025.csv',parse_dates=['timestamp_utc']).set_index('timestamp_utc').sort_index()
    rt=pd.read_csv(DATA/'rt15_all_development_2018_2021_2025.csv',parse_dates=['timestamp_utc']).set_index('timestamp_utc').sort_index()['RT15_price_usd_mwh']
    wind=pd.read_csv(DATA/'wind_5min_development_2025Dec_2026Jan.csv',parse_dates=['timestamp_utc']).set_index('timestamp_utc').sort_index()['HorseCreek_MW']
    return hourly,rt,wind

def clearing_asof(issue_utc):
    """Preserve original *assumed* DA publication policy, not a verified ERCOT timestamp."""
    local=pd.Timestamp(issue_utc).tz_convert(CHICAGO)
    last=local.normalize()+pd.Timedelta(hours=18)
    if last>local:last-=pd.Timedelta(days=1)
    return last.tz_convert('UTC')

def available_da(issue_utc,all_da):
    local=pd.Timestamp(issue_utc).tz_convert(CHICAGO)
    latest=local.date() if local.hour<18 else (local+pd.Timedelta(days=1)).date()
    return all_da.loc[all_da.index.tz_convert(CHICAGO).date<=latest]
