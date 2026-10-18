"""Prevent a recurrence of the observed zero-delta HL SoC bridge."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from run_development_day import soc_lookup
from run_multiday import local_midnight
from soc_handover import predict_delivery_soc
from ptes_mpc.physics import PlantParams

HERE=Path(__file__).resolve().parents[1]

def test_real_saved_prior_plan_has_nonzero_remaining_delta():
    raw=pd.read_csv(HERE/'baseline_reference'/'HL_2025-12-08_reference_fixture.csv')
    start=local_midnight('2025-12-08')
    bid=start+pd.Timedelta(hours=9,minutes=30)
    delivery=start+pd.Timedelta(hours=24)
    res=predict_delivery_soc(bidding_utc=bid,delivery_utc=delivery,
        observed_soc_mwhth=500.,previous_plan_start_utc=start,
        previous_reference_lookup=soc_lookup(start,0.,raw.hl_reference_mwhth.to_numpy()),
        capacity_mwhth=PlantParams().capacity_mwhth)
    assert res.previous_ref_at_bid==pytest.approx(608.2816693454472,abs=1e-6)
    assert res.previous_ref_at_delivery==pytest.approx(414.0138080454865,abs=1e-6)
    assert res.planned_change==pytest.approx(-194.2678612999607,abs=1e-6)
    assert res.predicted_at_delivery==pytest.approx(305.7321387000393,abs=1e-6)

def test_empty_still_logs_nonzero_raw_delta():
    start=local_midnight('2025-12-08')
    bid=start+pd.Timedelta(hours=9,minutes=30)
    f=lambda t:600. if t==bid else 400.
    res=predict_delivery_soc(bidding_utc=bid,delivery_utc=start+pd.Timedelta(days=1),
        observed_soc_mwhth=0.,previous_plan_start_utc=start,
        previous_reference_lookup=f,capacity_mwhth=1000.)
    assert res.planned_change == -200.
    assert res.predicted_before_bounds == -200.
    assert res.predicted_at_delivery == 0.
    assert res.clipped_by_bounds == 200.

@pytest.mark.parametrize('bad_bid', ['2025-12-07T00:00:00Z','2025-12-10T00:00:00Z'])
def test_invalid_issue_time_rejected(bad_bid):
    start=local_midnight('2025-12-08')
    with pytest.raises(ValueError):predict_delivery_soc(
        bidding_utc=pd.Timestamp(bad_bid),delivery_utc=start+pd.Timedelta(days=1),
        observed_soc_mwhth=100.,previous_plan_start_utc=start,
        previous_reference_lookup=lambda _:500.,capacity_mwhth=1000.)
