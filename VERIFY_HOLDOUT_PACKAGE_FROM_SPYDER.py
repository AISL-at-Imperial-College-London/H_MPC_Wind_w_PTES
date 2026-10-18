"""Fast preflight; no scored-week optimization or revenue evaluation."""
from pathlib import Path
import sys,hashlib,json
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from holdout_runtime import SEASONS,load_season,hourly_runtime_history
from wind_only.utils import local_midnight
from ptes_mpc.physics import PlantParams

def preflight(check_gurobi=True):
    rows=[]
    full=PlantParams()
    assert abs(full.capacity_mwhth-2685.633333333333)<1e-6
    half=full.capacity_mwhth/2.
    print(f'PTES capacity full = {full.capacity_mwhth:.6f} MWhth; half = {half:.6f} MWhth')
    for season,(day,a,b) in SEASONS.items():
        active=load_season(season)
        begin=local_midnight(day)
        first_issue=begin-pd.Timedelta(hours=14,minutes=30)
        history,n_bootstrap=hourly_runtime_history(active,first_issue)
        assert len(history)>=96 and n_bootstrap<=36
        assert (active['score_end']-active['score_start'])==pd.Timedelta(days=7)
        assert begin<=active['score_start']-pd.Timedelta(days=7)
        full_end=local_midnight((pd.Timestamp(day)+pd.Timedelta(days=15)).strftime('%Y-%m-%d'))
        assert full_end>=active['score_end']
        score_w=active['wind'].loc[(active['wind'].index>=active['score_start']) &
                                    (active['wind'].index<active['score_end'])]
        score_rt=active['rt'].loc[(active['rt'].index>=active['score_start']) &
                                  (active['rt'].index<active['score_end'])]
        score_da=active['hourly'].loc[(active['hourly'].index>=active['score_start']) &
                                      (active['hourly'].index<active['score_end'])]
        assert (len(score_w),len(score_rt),len(score_da))==(2016,672,168)
        print(f'{season:6s}: first bid {first_issue}; initial padding {n_bootstrap} h; '
              f'warmup {(active["score_start"]-begin)/pd.Timedelta(hours=1):.0f}h; '
              f'wind telemetry repaired {len(active["wind_repairs"])}; RT repaired {len(active["rt_repairs"])}')
        rows.append(dict(season=season,first_market_day=day,first_issue_utc=str(first_issue),
                         bootstrap_hours=n_bootstrap,score_start=a,score_end=b,
                         warmup_hours=float((active['score_start']-begin)/pd.Timedelta(hours=1)),
                         scored_wind_rows=2016,scored_DA_rows=168,scored_RT_rows=672,
                         wind_repair_count=len(active['wind_repairs']),
                         rt_repair_count=len(active['rt_repairs'])))
    paths=[HERE/'forecast_release'/'source_releases'/'horse_creek_forecasting_release'/'models'/'continuous_cvae_v3_checkpoint.pt',
           HERE/'forecast_release'/'ll_wind_xgb'/'direct_xgb_wind.json',
           HERE/'forecast_release'/'ll_rt_xgb'/'xgb_zero_delay_2025_trainJanJun.json']
    for path in paths:
        if not path.is_file():raise FileNotFoundError(str(path))
        print(path.relative_to(HERE), hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    # Checks only a PRE-2026 development origin, not protected holdout scores.
    sys.path.insert(0,str(HERE/'forecast_release'/'source_releases'/'horse_creek_forecasting_release'))
    from horse_creek_forecasts import JointForecastProvider
    s=JointForecastProvider('tcvae').generate('2025-12-08',n_raw=8,n_keep=3,seed=3714)
    assert s.wind_mw.shape==(3,84) and s.da_price_usd_mwh.shape==(3,84)
    assert (s.wind_mw>=0).all() and (s.wind_mw<=230).all()
    assert abs(float(s.weights.sum())-1)<1e-9
    assert s.metadata['method']=='continuous_CVAE_v3_calibrated'
    assert s.full_raw_wind_mw.shape[1] == s.metadata['pre_delivery_forecast_hours'] + 84
    assert s.full_timestamps_utc[-1] == s.timestamps_utc[-1]
    assert s.metadata['scenario_reduction'].startswith('original_algorithm_2')
    print('Continuous CVAE 96h history -> full forecast -> 84h HL reduced scenarios: OK')
    if check_gurobi:
        try:
            import gurobipy as gp
        except ImportError as ex:
            raise RuntimeError('Install Gurobi in the Python environment used by Spyder.') from ex
        with gp.Model() as m:
            m.Params.OutputFlag=0
            x=m.addVar(lb=0,ub=1,vtype=gp.GRB.BINARY)
            m.setObjective(x,gp.GRB.MAXIMIZE)
            m.optimize()
            assert m.Status==gp.GRB.OPTIMAL
        print('Gurobi license and binary solver: OK')
    print('PASS: data shape, 2026 deployment clock, model checkpoint and scenario settings.')
    return rows

if __name__=='__main__':
    preflight(True)
