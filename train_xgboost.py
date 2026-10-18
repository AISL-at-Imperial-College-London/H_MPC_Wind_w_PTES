#!/usr/bin/env python3
"""Train the two ACTIVE LL XGBoost forecast families in isolated output folders.

The historical source trainers have been restored byte-for-byte from V3.3.
This launcher supplies safe data/output defaults. It DOES NOT promote model
weights into the frozen MPC replay checkpoints.

Examples:
    python train_xgboost.py preflight
    python train_xgboost.py wind
    python train_xgboost.py rt --publication-lag 0
    python train_xgboost.py all
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
FORECAST = ROOT / 'forecast_release'
DEVELOPMENT = FORECAST / 'source_releases/horse_creek_forecasting_release/data'
WIND_SCRIPT = FORECAST / 'll_wind_xgb/train_direct_wind_xgb.py'
RT_SCRIPT = ROOT / 'training/xgboost/forecast_rt15_spread.py'
WIND_CSV = DEVELOPMENT / 'wind_5min_development_2025Dec_2026Jan.csv'
PROD_WIND = FORECAST / 'll_wind_xgb/direct_xgb_wind.json'
PROD_RT = FORECAST / 'll_rt_xgb/xgb_zero_delay_2025_trainJanJun.json'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def safe_folder(out):
    out=out.expanduser().resolve()
    if out==FORECAST.resolve() or FORECAST.resolve() in out.parents:
        raise ValueError('Refusing to write inside frozen forecast_release/ directory')
    if out==ROOT.resolve() or out==DEVELOPMENT.resolve():
        raise ValueError('Select a dedicated training output directory')
    return out


def preflight():
    import pandas as pd
    from xgboost import XGBRegressor
    for path in [WIND_CSV, DEVELOPMENT_HOURLY, DEVELOPMENT_RT, WIND_SCRIPT, RT_SCRIPT, PROD_WIND, PROD_RT]:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    w=pd.read_csv(WIND_CSV, usecols=['timestamp_utc','HorseCreek_MW'],parse_dates=['timestamp_utc'])
    w=w.set_index('timestamp_utc').sort_index()['HorseCreek_MW']
    start=pd.Timestamp('2025-12-05T06:00:00Z')
    end=pd.Timestamp('2026-01-29T00:00:00Z')
    if w.index.min()>start or w.index.max()<end-pd.Timedelta(minutes=5):
        raise RuntimeError('Bundled 5-minute wind development input is incomplete')
    mod=XGBRegressor();mod.load_model(PROD_WIND)
    if mod.n_features_in_!=48 or mod.get_booster().num_boosted_rounds()!=100:
        raise AssertionError('Unexpected production wind XGBoost architecture')
    rm=XGBRegressor();rm.load_model(PROD_RT)
    import importlib.util
    spec=importlib.util.spec_from_file_location('xgb_rt_features',RT_SCRIPT)
    rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
    runtime=FORECAST/'ll_rt_xgb/rt_xgb_runtime.py'
    s=importlib.util.spec_from_file_location('xgb_rt_runtime',runtime)
    r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
    if rt.COLS!=r.COLS or len(rt.COLS)!=rm.n_features_in_:
        raise AssertionError('RT15 model/trainer/runtime input features differ')
    if rm.get_booster().num_boosted_rounds()!=180:
        raise AssertionError('Unexpected production RT XGBoost rounds')
    print('OK: bundled wind and RT15 development CSV files are present')
    print('OK: frozen wind XGB has 48 inputs, 48 target increments, 100 rounds')
    print('OK: frozen RT XGB has 31 matching input features, 180 rounds')
    print('Training data and 2026 holdout data are kept separate')


DEVELOPMENT_HOURLY = DEVELOPMENT/'development_master_hourly_2018_2021_2024_2025.csv'
DEVELOPMENT_RT = DEVELOPMENT/'rt15_all_development_2018_2021_2025.csv'


def run_model(which, args):
    if which=='wind':
        out=safe_folder(args.output or ROOT/'training_runs/xgboost_wind')
        data=Path(args.wind_data).expanduser().resolve() if args.wind_data else WIND_CSV
        cmd=[sys.executable,'-u',str(WIND_SCRIPT),'--data',str(data),'--out',str(out),
             '--trees',str(args.wind_trees),'--stride',str(args.wind_stride)]
        inputs=[data];training_script=WIND_SCRIPT
    else:
        out=safe_folder(args.output or ROOT/'training_runs/xgboost_rt15')
        data=Path(args.data_dir).expanduser().resolve() if args.data_dir else DEVELOPMENT
        cmd=[sys.executable,'-u',str(RT_SCRIPT),'--data-dir',str(data),'--lag',
             str(args.publication_lag),'--output',str(out)]
        inputs=[data/'rt15_all_development_2018_2021_2025.csv',
                data/'development_master_hourly_2018_2021_2024_2025.csv']
        training_script=RT_SCRIPT
    for x in inputs+[training_script]:
        if not x.is_file(): raise FileNotFoundError(str(x))
    if which=='rt' and args.publication_lag!=0:
        print('WARNING: nonzero RT publication lag is a separate, more conservative model; not the frozen MPC baseline.',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    print('Executing:',subprocess.list2cmdline(cmd),flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    manifest={'family':which,'source_script':str(training_script.relative_to(ROOT)),
        'script_sha256':sha(training_script),'inputs':{str(p):sha(p) for p in inputs},
        'invocation':cmd,'production_checkpoint_modified':False,
        'notes':'New training outputs are NOT automatically loaded by the existing MPC replay.'}
    if which=='rt':
        manifest['assumed_rt_publication_lag_minutes']=args.publication_lag
        manifest['frozen_baseline_rt_assumption']='zero-minute publication delay'
        manifest['output_model_for_live_runtime']='XGB_RT_spread_development_2025.json'
    (out/'TRAINING_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('Training outputs:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=['preflight','wind','rt','all'])
    p.add_argument('--output',type=Path,help='Isolated output directory; only valid for wind or rt')
    p.add_argument('--wind-data',help='Alternate genuine 5-min development CSV')
    p.add_argument('--wind-trees',type=int,default=100)
    p.add_argument('--wind-stride',type=int,default=3)
    p.add_argument('--data-dir',type=Path,help='Alternative RT/DA/wind development tables directory')
    p.add_argument('--publication-lag',type=int,default=0,help='Assumed minutes after RT interval end; frozen model uses 0')
    a=p.parse_args()
    if a.wind_trees<1 or a.wind_stride<1:p.error('wind trees and stride must be positive')
    if a.publication_lag<0 or a.publication_lag%15:
        p.error('RT publication lag must be a nonnegative multiple of 15 minutes')
    if a.model=='all' and a.output is not None:
        p.error('Use default separate outputs for --model all')
    if a.model=='preflight':preflight();return 0
    for model in (['wind','rt'] if a.model=='all' else [a.model]):
        run_model(model,a)
    return 0

if __name__=='__main__':sys.exit(main())
