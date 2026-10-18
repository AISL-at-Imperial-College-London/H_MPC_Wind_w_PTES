"""Preflight for the combined V3 supplementary 3x3x2 campaign."""
from __future__ import annotations
from pathlib import Path
import hashlib
import pandas as pd
from extra_v3_runtime import SEASONS,load_season,history_available_at_bid,expected_market_days
from ptes_mpc.physics import PlantParams

HERE=Path(__file__).resolve().parent

def preflight(check_gurobi=True):
    full=PlantParams()
    assert abs(full.capacity_mwhth-2685.633333333333)<1e-5
    print(f'Plant: full={full.capacity_mwhth:.4f} MWhth; half={full.capacity_mwhth*.5:.4f} MWhth; same charge/discharge MW ratings',flush=True)
    for season,(first,start,end) in SEASONS.items():
        active=load_season(season)
        begin=pd.Timestamp(first).tz_localize('America/Chicago').tz_convert('UTC')
        issue=begin-pd.Timedelta(hours=14,minutes=30)
        history,pad=history_available_at_bid(active,issue)
        assert len(history)==96 and 0<=pad<=95 and history.index.max()<issue
        assert (active['score_end']-active['score_start'])==pd.Timedelta(days=7)
        assert begin<active['score_start']
        count=expected_market_days(season)
        final_local=(pd.Timestamp(first)+pd.Timedelta(days=count)).strftime('%Y-%m-%d')
        final=pd.Timestamp(final_local).tz_localize('America/Chicago').tz_convert('UTC')
        assert final>=active['score_end']
        a,b=active['score_start'],active['score_end']
        assert len(active['wind'].loc[a:b-pd.Timedelta(minutes=5)])==2016
        assert len(active['rt'].loc[a:b-pd.Timedelta(minutes=15)])==672
        assert len(active['hourly'].loc[a:b-pd.Timedelta(hours=1)])==168
        print(f'{season:11s} {start[:10]}..{(pd.Timestamp(end)-pd.Timedelta(days=1)).strftime("%Y-%m-%d")} | {count} local days | {float((a-begin)/pd.Timedelta(hours=1)):.0f}h warmup | {pad}h prior history padded | wind repairs {len(active["wind_repairs"])}',flush=True)
    ckpts=[
     'forecast_release/source_releases/horse_creek_forecasting_release/models/continuous_cvae_v3_checkpoint.pt',
     'forecast_release/source_releases/horse_creek_forecasting_release/models/continuous_calibration_residual_paths.npz',
     'forecast_release/ll_wind_xgb/direct_xgb_wind.json',
     'forecast_release/ll_rt_xgb/xgb_zero_delay_2025_trainJanJun.json',
    ]
    for rel in ckpts:
        p=HERE/rel
        if not p.is_file():raise FileNotFoundError(p)
        print('Checkpoint OK:',rel,hashlib.sha256(p.read_bytes()).hexdigest()[:12],flush=True)
    if check_gurobi:
        import gurobipy as gp
        with gp.Model() as m:
            m.Params.OutputFlag=0
            x=m.addVar(lb=0,ub=1,vtype=gp.GRB.BINARY)
            m.setObjective(x,gp.GRB.MAXIMIZE);m.optimize()
            assert m.Status==gp.GRB.OPTIMAL
        print('Gurobi binary MIP/license: OK',flush=True)
    print('PRE-FLIGHT PASS — V3 extra-week data and model checkpoints (not a full campaign solve).',flush=True)

if __name__=='__main__':preflight(True)
