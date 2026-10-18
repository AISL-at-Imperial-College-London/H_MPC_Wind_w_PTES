"""Continuous v3 tests; no holdout outcomes exposed."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'forecast_release/source_releases/horse_creek_forecasting_release'))
from horse_creek_forecasts import JointForecastProvider,MODEL,CALIBRATION

def test_continuous_500_to_50():
    assert MODEL.name=='continuous_cvae_v3_checkpoint.pt' and CALIBRATION.exists()
    s=JointForecastProvider('tcvae').generate('2025-12-08',n_raw=500,n_keep=50,seed=3714)
    assert s.wind_mw.shape==(50,84) and s.raw_wind_mw.shape==(500,84)
    assert s.full_raw_wind_mw.shape==(500,99)
    assert s.full_timestamps_utc[0]==pd.Timestamp('2025-12-07T15:00:00Z')
    assert s.full_timestamps_utc[-1]==s.timestamps_utc[-1]
    assert np.array_equal(s.full_raw_wind_mw[:,-84:],s.raw_wind_mw)
    assert np.array_equal(s.full_reduced_wind_mw[:,-84:],s.wind_mw)
    assert np.isclose(s.weights.sum(),1.)
    assert s.metadata['reduction_input_hours']==84
    assert s.metadata['no_2026_training_or_calibration']

def test_seed_reproducibility():
    p=JointForecastProvider('tcvae')
    a=p.generate('2025-12-08',n_raw=8,n_keep=3,seed=3714)
    b=p.generate('2025-12-08',n_raw=8,n_keep=3,seed=3714)
    assert np.array_equal(a.full_raw_wind_mw,b.full_raw_wind_mw)
    assert np.array_equal(a.full_raw_da_price_usd_mwh,b.full_raw_da_price_usd_mwh)
    assert np.array_equal(a.weights,b.weights)
