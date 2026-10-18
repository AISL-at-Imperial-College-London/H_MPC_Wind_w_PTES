"""Causal, instrumented bid-to-delivery state-of-charge prediction.

The previous HL plan is explicitly provided by the caller. No global plan
lookup or future actual observation is allowed. This bridge is a planning
heuristic, not a prediction of future lower-level MPC actions.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SoCHandover:
    observed_at_bid: float
    previous_ref_at_bid: float
    previous_ref_at_delivery: float
    planned_change: float
    predicted_before_bounds: float
    predicted_at_delivery: float
    clipped_by_bounds: float


def predict_delivery_soc(*, bidding_utc, delivery_utc, observed_soc_mwhth,
                         previous_plan_start_utc, previous_reference_lookup,
                         capacity_mwhth):
    """Predict delivery SoC from PREVIOUS plan at D-1 bid time.

    If the planned remaining change is negative and observed tank is empty,
    clipping can legitimately yield a zero estimate. Record RAW delta as well
    so this is distinguishable from a broken reference lookup.
    """
    bid=pd.Timestamp(bidding_utc)
    delivery=pd.Timestamp(delivery_utc)
    start=pd.Timestamp(previous_plan_start_utc)
    if bid.tz is None or delivery.tz is None or start.tz is None:
        raise ValueError('all SoC handover timestamps must be timezone aware')
    bid=bid.tz_convert('UTC');delivery=delivery.tz_convert('UTC');start=start.tz_convert('UTC')
    if not (start <= bid < delivery and delivery <= start + pd.Timedelta(hours=84)):
        raise ValueError('handover times lie outside the previous forecast horizon')
    if not np.isfinite(observed_soc_mwhth) or not (0 <= observed_soc_mwhth <= capacity_mwhth):
        raise ValueError('invalid observed SoC')
    old_now=float(previous_reference_lookup(bid))
    old_delivery=float(previous_reference_lookup(delivery))
    if not np.isfinite(old_now) or not np.isfinite(old_delivery):
        raise ValueError('previous HL reference contains nonfinite inventory')
    if not (0<=old_now<=capacity_mwhth+1e-4 and 0<=old_delivery<=capacity_mwhth+1e-4):
        raise ValueError('old HL reference outside tank bounds')
    delta=old_delivery-old_now
    raw=float(observed_soc_mwhth+delta)
    predicted=float(np.clip(raw,0.,capacity_mwhth))
    return SoCHandover(float(observed_soc_mwhth),old_now,old_delivery,delta,
                       raw,predicted,predicted-raw)
