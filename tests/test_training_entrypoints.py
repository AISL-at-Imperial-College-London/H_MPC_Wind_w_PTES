"""Training is separate from frozen MPC inference and uses pre-2026 data."""
from pathlib import Path
import importlib.util
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'forecast_release/source_releases/horse_creek_forecasting_release/data/development_master_hourly_2018_2021_2024_2025.csv'


def _trainer():
    path = ROOT / 'training/thesis_tcvae/train_thesis_tcvae.py'
    spec = importlib.util.spec_from_file_location('repo_thesis_tcvae_trainer', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_data_safe_time_splits_and_pandas3_compatible():
    t = _trainer()
    df, files = t.load_hourly_source(DATA)
    assert len(files) == 1
    assert df.timestamp_utc.dt.year.max() == 2025
    df, policy = t.prepare_splits(df)
    assert 'validate 2024' in policy
    starts, counts = t.eligible_starts(df)
    assert counts['train'] > 10000
    assert counts['validation'] > 100
    assert counts['test'] > 100
    assert (df.iloc[starts].split.to_numpy() == 'unused').sum() == 0


def test_cli_help_exposes_separate_training_and_calibration():
    result = subprocess.run([sys.executable, str(ROOT/'train_forecasts.py'), '--help'],
                            capture_output=True, text=True, check=True)
    for x in ('thesis', 'check', 'continuous', 'calibrate'):
        assert x in result.stdout


def test_production_ckpt_not_overwritten_by_training_test():
    import json, hashlib
    manifest = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    path = 'forecast_release/source_releases/horse_creek_forecasting_release/models/continuous_cvae_v3_checkpoint.pt'
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == manifest['source_files_sha256'][path]
