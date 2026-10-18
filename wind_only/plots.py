"""A few comparison-ready wind-only figures, saved without interactive GUI."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def make_plots(folder,events,commitments,forecasts,daily,quarters):
    folder=Path(folder)
    e=events.copy();e['time_utc']=pd.to_datetime(e.time_utc,utc=True)
    q=commitments.copy();q['hour_start_utc']=pd.to_datetime(q.hour_start_utc,utc=True)
    def save(name):
        plt.tight_layout();plt.savefig(folder/name,dpi=155,bbox_inches='tight');plt.close()
    fig,ax=plt.subplots(2,1,figsize=(14,8),sharex=True)
    ax[0].plot(e.time_utc,e.available_wind_mw,label='Available wind',linewidth=.9)
    ax[0].plot(e.time_utc,e.dispatched_wind_mw,label='Realized dispatched wind',linewidth=.8)
    ax[0].step(q.hour_start_utc,q.DA_commitment_MW,where='post',label='Wind-only HL DA position',alpha=.8)
    ax[0].set(ylabel='MW',title='Wind-only: available wind, dispatched wind, HL day-ahead bids')
    ax[0].legend()
    ax[1].fill_between(e.time_utc,e.curtailed_wind_mw,alpha=.5,label='Curtailed wind (MW)')
    ax[1].plot(e.time_utc,e.wind_dispatch_cap_mw,label='Commanded wind output cap (MW)',alpha=.65,linewidth=.7)
    ax[1].set(ylabel='MW',title='Wind curtailment and output cap')
    ax[1].legend();save('01_wind_dispatch_and_DA.png')
    fig,ax=plt.subplots(2,1,figsize=(14,8),sharex=True)
    ax[0].plot(e.time_utc,e.actual_rt15_usd_mwh,label='Realized RT price')
    ax[0].plot(e.time_utc,e.forecast_rt15_usd_mwh,label='RT forecast',alpha=.65)
    ax[0].set(ylabel='USD/MWh',title='Causal RT forecasting and settlement price');ax[0].legend()
    ax[1].plot(e.time_utc,e.curtailed_wind_mw,label='Curtailed wind MW')
    ax[1].set(ylabel='MW',title='Curtailment relative to RT prices');ax[1].legend()
    save('02_rt_price_and_curtailment.png')
    fig,ax=plt.subplots(figsize=(12,5))
    ax.bar(daily.delivery_day_local,daily.total_revenue_USD,label='DA+RT revenue')
    ax.set(ylabel='USD',title='Wind-only combined DA/RT realized daily cashflow')
    plt.xticks(rotation=25);save('03_daily_revenue.png')
    fig,ax=plt.subplots(figsize=(12,5))
    ax.step(quarters.quarter_start_utc,quarters.DA_energy_deviation_mwh,where='post',label='15min physical energy − DA schedule')
    ax.axhline(0,color='grey',linewidth=.7)
    ax.set(ylabel='MWh per 15min',title='Wind-only settled energy deviations')
    ax.legend();save('04_quarterly_imbalances.png')
