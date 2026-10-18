#!/usr/bin/env python3
"""Safe, explicit entry point for training two INCOMPATIBLE forecasting models.

Never modifies production checkpoints in forecast_release/.../models.
Run from any current working directory; intended for CLI and Spyder.
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
FORECAST=ROOT/'forecast_release'/'source_releases'/'horse_creek_forecasting_release'
DEFAULT_DATA=FORECAST/'data'/'development_master_hourly_2018_2021_2024_2025.csv'


def command(args):
    data=Path(args.input).expanduser().resolve() if args.input else DEFAULT_DATA
    if not data.is_file():
        raise FileNotFoundError(f'Missing development training input: {data}')
    if args.model in ('continuous','calibrate') and data.suffix.lower()!='.csv':
        raise ValueError('Continuous CVAE training/calibration requires an hourly development CSV')
    out=(Path(args.output).expanduser().resolve() if args.output else
         ROOT/'training_runs'/('thesis_tcvae' if args.model in ('thesis','check') else 'continuous_cvae'))
    if args.model=='thesis':
        script=ROOT/'training'/'thesis_tcvae'/'train_thesis_tcvae.py'
        epochs=args.epochs if args.epochs is not None else (2 if args.smoke else 250)
        batch=args.batch if args.batch is not None else 32
        cmd=[sys.executable,'-u',str(script),'--input',str(data),'--output',str(out),
             '--epochs',str(epochs),'--batch',str(batch),'--threads',str(args.threads),
             '--device',args.device,'--seed',str(args.seed if args.seed is not None else 20261017)]
        if args.smoke:cmd+=['--max-examples','64']
    elif args.model=='check':
        cp=Path(args.checkpoint).expanduser().resolve() if args.checkpoint else out/'best_checkpoint.pt'
        if not cp.is_file():raise FileNotFoundError(f'No thesis checkpoint: {cp}')
        script=ROOT/'training'/'thesis_tcvae'/'check_conditioning.py'
        cmd=[sys.executable,'-u',str(script),'--input',str(data),'--checkpoint',str(cp),
             '--output',str(out/'conditioning_check'),'--day',args.day,
             '--dates',str(args.dates),'--scenarios',str(args.scenarios),'--device',args.device]
    elif args.model=='continuous':
        script=FORECAST/'train_continuous_cvae.py'
        epochs=args.epochs if args.epochs is not None else (1 if args.smoke else 80)
        batch=args.batch if args.batch is not None else (256 if not args.smoke else 64)
        cmd=[sys.executable,'-u',str(script),'--input',str(data),'--out',str(out),
             '--epochs',str(epochs),'--batch',str(batch),'--threads',str(args.threads),
             '--stride',str(96 if args.smoke else args.stride),
             '--seed',str(args.seed if args.seed is not None else 2600)]
    elif args.model=='calibrate':
        cp=Path(args.checkpoint).expanduser().resolve() if args.checkpoint else out/'continuous_cvae_v3_checkpoint.pt'
        if not cp.is_file():raise FileNotFoundError(f'No continuous checkpoint: {cp}')
        script=FORECAST/'calibrate_continuous_cvae.py'
        cmd=[sys.executable,'-u',str(script),'--input',str(data),'--checkpoint',str(cp),
             '--out',str(out),'--threads',str(args.threads)]
    else:raise ValueError(args.model)
    if out.resolve()==(FORECAST/'models').resolve():
        raise ValueError('Production checkpoint folder is protected; choose another --output')
    out.mkdir(parents=True,exist_ok=True)
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=['thesis','check','continuous','calibrate'],
                   help='thesis=96h thesis trainer; check=thesis conditioning; continuous=active MPC model trainer; calibrate=active MPC residual calibration')
    p.add_argument('--input',help='Development CSV (default: repository development master). Thesis also accepts historical ZIP')
    p.add_argument('--output',help='Isolated experiment output; default: training_runs/<model>')
    p.add_argument('--checkpoint',help='Explicit checkpoint for check/calibrate')
    p.add_argument('--epochs',type=int,help='Number of training epochs')
    p.add_argument('--batch',type=int,help='Training batch size')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto',
                   help='Thesis model and diagnostics only; continuous defaults to its original CPU trainer')
    p.add_argument('--seed',type=int)
    p.add_argument('--stride',type=int,default=3,help='Continuous training origin stride')
    p.add_argument('--smoke',action='store_true',help='Tiny smoke test, not a trained forecasting model')
    p.add_argument('--day',default='2025-07-14',help='Thesis conditioning example date')
    p.add_argument('--dates',type=int,default=48,help='Thesis conditioning diagnostic dates')
    p.add_argument('--scenarios',type=int,default=64)
    args=p.parse_args()
    if args.smoke and args.model not in ('thesis','continuous'):
        p.error('--smoke only applies to thesis or continuous training')
    if args.epochs is not None and args.epochs<1:p.error('--epochs must be positive')
    cmd=command(args)
    print('Dataset:',args.input or DEFAULT_DATA,flush=True)
    print('Python:',sys.executable,flush=True)
    print('Executing:', ' '.join('"'+c+'"' if ' ' in c else c for c in cmd),flush=True)
    return subprocess.call(cmd,cwd=ROOT)

if __name__=='__main__':sys.exit(main())
