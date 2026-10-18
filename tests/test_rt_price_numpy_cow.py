"""RT-price forecast array must be writable under pandas Copy-on-Write."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'forecast_release' / 'll_rt_xgb'))
from rt_xgb_runtime import RT15XGBoostProvider, COLS


def test_rt_prediction_cow_writable_and_does_not_modify_da_baseline():
    # The installed Windows pandas build can return a read-only NumPy view.
    # Forcing Copy-on-Write reproduces that behaviour on older versions too.
    issue = pd.Timestamp('2025-12-18T06:00:00Z')
    targets = pd.date_range(issue, periods=16, freq='15min', tz='UTC')
    X = pd.DataFrame(0., index=range(16), columns=COLS)
    X['da_target'] = np.arange(16, dtype=float) + 100.
    expected_da = X['da_target'].to_numpy(dtype=float, copy=True)
    same = np.array([True] * 12 + [False] * 4, dtype=bool)
    predictor = RT15XGBoostProvider.__new__(RT15XGBoostProvider)
    predictor.model = SimpleNamespace(predict=lambda input_matrix: np.full(len(input_matrix), 5.0))

    with pd.option_context('mode.copy_on_write', True):
        assert not X['da_target'].to_numpy(dtype=float).flags.writeable
        with patch.object(predictor, '_inputs', return_value=(issue, targets, X, same)):
            res = predictor.forecast(issue, None, None, None,
                                     da_cleared_at_utc=issue - pd.Timedelta(hours=1),
                                     method='xgb')
            expected = expected_da + np.r_[np.full(12, 5.0), np.zeros(4)]
            np.testing.assert_allclose(res['RT15_price_USD_MWh'], expected)
            # No changes to the input feature frame.
            np.testing.assert_allclose(X['da_target'].to_numpy(dtype=float), expected_da)
            da_only = predictor.forecast(issue, None, None, None,
                                        da_cleared_at_utc=issue - pd.Timedelta(hours=1),
                                        method='da_only')
            np.testing.assert_allclose(da_only['RT15_price_USD_MWh'], expected_da)
            assert res['RT15_price_USD_MWh'].flags.writeable
