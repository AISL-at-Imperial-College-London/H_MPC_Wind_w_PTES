"""Regression for the date-unit bug seen in the Windows seven-day run."""
import numpy as np
import pandas as pd
import pytest
import run_development_day
from run_development_day import soc_lookup
from soc_handover import predict_delivery_soc


def test_datetime_microsecond_queries_do_not_flatten_soc():
    start = pd.Timestamp('2025-12-08T06:00:00Z')
    # distinct values all the way to hour 84, so the terminal value is recognizable
    references = np.arange(1, 85, dtype=float) * 7.5
    lookup = soc_lookup(start, 0, references)
    for unit in ('ns', 'us', 'ms'):
        dt = (start + pd.Timedelta(hours=9, minutes=30)).as_unit(unit)
        assert lookup(dt) == pytest.approx(9.5 * 7.5, abs=1e-8)
        assert lookup((start + pd.Timedelta(hours=24)).as_unit(unit)) == pytest.approx(24*7.5, abs=1e-8)
        assert lookup(dt) != pytest.approx(references[-1])


def test_old_us_datetimeindex_bug_reproduces_prior_result():
    start=pd.Timestamp('2025-12-08T06:00:00Z')
    times=pd.date_range(start,periods=85,freq='h',tz='UTC').as_unit('us')
    vals=np.arange(85, dtype=float)
    # This is exactly the old mixing error: index ticks in microseconds,
    # Timestamp.value always in nanoseconds, so interpolation returns terminal.
    wrong=np.interp((start + pd.Timedelta(hours=9,minutes=30)).value,
                    times.asi8.astype(float),vals)
    assert wrong == vals[-1]
    assert soc_lookup(start,0.,vals[1:])(start+pd.Timedelta(hours=9,minutes=30)) == pytest.approx(9.5)


def test_saved_december_reference_bridge_nonzero_across_units():
    from pathlib import Path
    f=Path(__file__).resolve().parents[1]/'baseline_reference'/'HL_2025-12-08_reference_fixture.csv'
    ref=pd.read_csv(f).hl_reference_mwhth.to_numpy()
    start=pd.Timestamp('2025-12-08T06:00:00Z')
    lookup=soc_lookup(start,0.,ref)
    for unit in ('ns','us'):
        bridge=predict_delivery_soc(
            bidding_utc=(start+pd.Timedelta(hours=9,minutes=30)).as_unit(unit),
            delivery_utc=(start+pd.Timedelta(days=1)).as_unit(unit),
            observed_soc_mwhth=500., previous_plan_start_utc=start.as_unit(unit),
            previous_reference_lookup=lookup, capacity_mwhth=2685.633333333333)
        assert bridge.previous_ref_at_bid == pytest.approx(608.2816693454472, abs=1e-6)
        assert bridge.previous_ref_at_delivery == pytest.approx(414.0138080454865, abs=1e-6)
        assert bridge.planned_change == pytest.approx(-194.2678612999607, abs=1e-6)
