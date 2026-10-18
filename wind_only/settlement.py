"""ERCOT style two-settlement signed cash-flow identity (no SCED revenue)."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict

@dataclass
class SettlementLedger:
    # All times UTC and 5min start aligned; DA commitments keyed by the UTC hour start.
    q_da_mw: dict
    lambda_da: dict
    actual_rt_prices: dict
    energy_mwh: dict = field(default_factory=lambda:defaultdict(float))
    count_5min: dict = field(default_factory=lambda:defaultdict(int))
    seen: set = field(default_factory=set)

    @staticmethod
    def bucket(t,minutes):
        if t.tzinfo is None: raise ValueError('timezone-aware UTC timestamps required')
        t=t.astimezone(timezone.utc)
        return t.replace(minute=(t.minute//minutes)*minutes,second=0,microsecond=0)

    def record(self, t, physical_poi_mw, dt_hr=1/12):
        if t in self.seen: raise ValueError('duplicate five-minute interval')
        if t.second or t.microsecond or t.minute%5: raise ValueError('unaligned five-minute interval')
        k=self.bucket(t,15)
        if self.count_5min[k]>=3: raise ValueError('too many intervals in settlement bucket')
        self.seen.add(t)
        self.energy_mwh[k]+=physical_poi_mw*dt_hr
        self.count_5min[k]+=1

    def quarter_settlement(self,k):
        if self.count_5min[k]!=3: raise ValueError('incomplete RT quarter')
        h=self.bucket(k,60)
        q=self.q_da_mw[h]
        return self.actual_rt_prices[k]*(self.energy_mwh[k] - q/4.)

    def full_cashflow(self):
        hours={self.bucket(k,60) for k in self.energy_mwh}
        for h in hours:
            if any(self.count_5min[h+timedelta(minutes=15*i)]!=3 for i in range(4)):
                raise ValueError('incomplete DA hour')
        da=sum(self.q_da_mw[h]*self.lambda_da[h] for h in hours)
        rt=sum(self.quarter_settlement(k) for k in self.energy_mwh)
        return {'da_usd': da, 'rt_usd': rt, 'total_usd': da+rt,
                'phys_mwh': sum(self.energy_mwh.values()),
                'scheduled_mwh':sum(self.q_da_mw[h] for h in hours)}
