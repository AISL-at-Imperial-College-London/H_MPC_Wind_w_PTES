"""Spyder F5 launcher for the calibrated continuous CVAE used by MPC.

Runs training, then calibration from its new checkpoint. Does not alter
production frozen MPC artifacts. Set SMOKE_TEST_ONLY=True for installation
check, which skips calibration and is NOT a forecasting model.
"""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).resolve().parent
SMOKE_TEST_ONLY=False
EPOCHS=80
OUTPUT=ROOT/'training_runs'/('continuous_smoke' if SMOKE_TEST_ONLY else 'continuous_cvae')
command=[sys.executable,'-u',str(ROOT/'train_forecasts.py'),'continuous','--output',str(OUTPUT),
         '--epochs',str(1 if SMOKE_TEST_ONLY else EPOCHS)]
if SMOKE_TEST_ONLY:command+=['--smoke']
status=subprocess.call(command,cwd=ROOT)
if status:raise RuntimeError(f'Continuous CVAE training failed (exit {status})')
if not SMOKE_TEST_ONLY:
    command=[sys.executable,'-u',str(ROOT/'train_forecasts.py'),'calibrate','--output',str(OUTPUT)]
    status=subprocess.call(command,cwd=ROOT)
    if status:raise RuntimeError(f'Continuous CVAE calibration failed (exit {status})')
print('Continuous model outputs:',OUTPUT,flush=True)
