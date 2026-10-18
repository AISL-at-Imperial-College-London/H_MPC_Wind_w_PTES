from unittest.mock import patch
import matplotlib.pyplot as plt
from run_multiday import _force_spyder_inline_for_display


def test_restore_inline_after_forecasting_module_reset():
    old=plt.get_backend()
    try:
        plt.switch_backend('Agg')
        assert str(plt.get_backend()).lower()=='agg'
        with patch('IPython.get_ipython', return_value=None):
            _force_spyder_inline_for_display()
        assert 'inline' in plt.get_backend().lower()
    finally:
        plt.switch_backend(old)
