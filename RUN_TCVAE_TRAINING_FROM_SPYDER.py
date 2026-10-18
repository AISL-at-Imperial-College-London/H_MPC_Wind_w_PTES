"""Spyder F5 launcher; trains the 96h/96h thesis T-CVAE.

Uses canonical 2018-2025 development data *already in this repository*.
First set SMOKE_TEST_ONLY=True to verify setup; restore False for full run.
Model outputs go into gitignored training_runs/; MPC checkpoint stays frozen.
"""
from pathlib import Path
import subprocess,sys

ROOT=Path(__file__).resolve().parent
SMOKE_TEST_ONLY=False
EPOCHS=250
BATCH=32
DATASET=None # None = repository development CSV. Set full CSV/ZIP path to override.
OUTPUT=ROOT/'training_runs'/('thesis_smoke' if SMOKE_TEST_ONLY else 'thesis_tcvae')
cmd=[sys.executable,'-u',str(ROOT/'train_forecasts.py'),'thesis','--output',str(OUTPUT),
     '--epochs',str(2 if SMOKE_TEST_ONLY else EPOCHS),'--batch',str(BATCH)]
if SMOKE_TEST_ONLY:cmd+=['--smoke']
if DATASET is not None:cmd+=['--input',str(DATASET)]
print('Running thesis T-CVAE training in:',sys.executable,flush=True)
status=subprocess.call(cmd,cwd=ROOT)
if status:raise RuntimeError(f'T-CVAE training failed (exit status {status})')
print('Training finished; checkpoint and metrics in:',OUTPUT,flush=True)
