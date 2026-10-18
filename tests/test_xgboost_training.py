"""Reproducibility checks for the active low-level XGBoost forecasters.

Tests do not retrain production models or include holdout labels in learning.
"""
from pathlib import Path
import importlib.util
import subprocess
import sys
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    x=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(x)
    return x


def test_cli_and_model_freeze():
    mod=module('xgb_safe_launcher','train_xgboost.py')
    with pytest.raises(ValueError,match='frozen'):
        mod.safe_folder(ROOT/'forecast_release/ll_rt_xgb/new_model')
    m=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())['source_files_sha256']
    for name in ('forecast_release/ll_rt_xgb/xgb_zero_delay_2025_trainJanJun.json',
                 'forecast_release/ll_wind_xgb/direct_xgb_wind.json'):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==m[name]


def test_wind_training_strict_time_split_and_48_output():
    t=module('original_wind_trainer','forecast_release/ll_wind_xgb/train_direct_wind_xgb.py')
    w=pd.read_csv(ROOT/'forecast_release/source_releases/horse_creek_forecasting_release/data/wind_5min_development_2025Dec_2026Jan.csv',
                  parse_dates=['timestamp_utc']).set_index('timestamp_utc')['HorseCreek_MW']
    train,val=t.make_windows(w,train_stride=30,valid_stride=8)
    assert len(train[0])>50 and len(val[0])>10
    assert train[1].shape[1]==48 and train[2].shape[1]==48
    assert max(train[0])+pd.Timedelta(hours=4)<=t.TRAIN_END
    assert min(val[0])>=t.TRAIN_END
    assert max(val[0])+pd.Timedelta(hours=4)<=t.VALID_END
    assert val[3].shape[0]==val[2].shape[0]


def test_rt_feature_schema_and_causal_training_targets():
    original=module('original_rt_training','training/xgboost/forecast_rt15_spread.py')
    prod=module('production_rt_runtime','forecast_release/ll_rt_xgb/rt_xgb_runtime.py')
    assert original.COLS==prod.COLS
    assert len(original.COLS)==31
    development=ROOT/'forecast_release/source_releases/horse_creek_forecasting_release/data'
    rt,da,wind=original.base_tables(development)
    # A 5-day slice deliberately too short for full RT fitting, enough to test
    # original source feature creation and lag semantics without expensive training.
    frame=original.make_table(rt.loc['2025-02-01':'2025-02-05'],
                              da.loc['2025-01-31':'2025-02-06'],
                              wind.loc['2025-01-31':'2025-02-06'],lag_min=0)
    assert not frame.empty
    assert (frame.target_utc>=frame.issue_utc).all()
    assert (frame['split']=='train').all()
    assert np.isfinite(frame[original.COLS].to_numpy(dtype=float)).all()
    first=frame.iloc[0]
    assert np.isclose(first['rt_lag_0'],rt.loc[first['issue_utc']-pd.Timedelta(minutes=15)])
    assert first['lead_min']>=15
    assert all(first[original.COLS].notna())
