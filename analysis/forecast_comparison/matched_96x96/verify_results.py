#!/usr/bin/env python3
"""Validate saved matched 96/96 metrics and causal date-window metadata."""
from pathlib import Path
import pandas as pd
import numpy as np

def verify(result_folder):
    folder=Path(result_folder)
    daily=pd.read_csv(folder/'daily_metrics.csv')
    weekly=pd.read_csv(folder/'weekly_metrics.csv')
    pooled=pd.read_csv(folder/'pooled_metrics.csv')
    total=pd.read_csv(folder/'multi_seed_metrics.csv')
    audit=pd.read_csv(folder/'issuance_audit.csv')
    expected_seeds={3714,13714,2718}
    assert set(daily.seed)==expected_seeds
    assert set(daily.method)=={'copula_96x96','deployed_cvae_96x96_eval'}
    assert set(daily.ensemble)=={'raw_500','reduced_50'}
    assert len(daily)==3*3*4*2*2
    assert len(audit)==3*3*4*2
    assert set(audit.num_input_hours)=={96} and set(audit.num_target_hours)=={96}
    latest=pd.to_datetime(audit.last_known_hour_start_utc,utc=True)
    first=pd.to_datetime(audit.first_forecast_hour_utc,utc=True)
    last=pd.to_datetime(audit.last_forecast_hour_utc,utc=True)
    cutoff=pd.to_datetime(audit.cutoff_utc,utc=True)
    assert ((first-latest)==pd.Timedelta(hours=1)).all()
    assert ((last-first)==pd.Timedelta(hours=95)).all()
    assert ((latest+pd.Timedelta(hours=1))<=cutoff).all()
    assert ((first==cutoff.dt.floor('h'))).all()
    cmp=daily.groupby(['seed','ensemble','method'],as_index=False)[['wind_crps','price_crps','energy_score','wind_cover90','price_cover90']].mean()
    got=pooled.merge(cmp,on=['seed','ensemble','method'],suffixes=('_saved','_computed'))
    for metric in ['wind_crps','price_crps','energy_score','wind_cover90','price_cover90']:
        assert np.allclose(got[f'{metric}_saved'],got[f'{metric}_computed'],rtol=0,atol=1e-12)
    avg=total[total.ensemble=='reduced_50'].set_index('method')
    c=avg.loc['copula_96x96'];v=avg.loc['deployed_cvae_96x96_eval']
    assert v.price_crps<c.price_crps and v.energy_score<c.energy_score
    print(f'PASS: {len(audit)} causal forecast runs, {len(daily)} score records, 3 seeds, 12 origins, 96/96 matching and aggregate consistency.')
if __name__=='__main__':
    import sys
    verify(sys.argv[1] if len(sys.argv)>1 else 'results_96x96')
