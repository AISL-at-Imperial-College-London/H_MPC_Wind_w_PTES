"""Spyder F5 launcher: choose model and run locally. New checkpoints are isolated.

IMPORTANT: full runs may be computationally expensive; no MPC checkpoints overwritten.
"""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
MODEL='all'  # 'wind', 'rt', 'all', or 'preflight'
RT_PUBLICATION_LAG_MINUTES=0  # matches frozen RT-price forecast checkpoint

if __name__=='__main__':
    cmd=[sys.executable,'-u',str(ROOT/'train_xgboost.py'),MODEL,
         '--publication-lag',str(RT_PUBLICATION_LAG_MINUTES)]
    subprocess.run(cmd,cwd=ROOT,check=True)
