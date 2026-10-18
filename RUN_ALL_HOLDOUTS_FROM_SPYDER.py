"""ONE-SHOT protected 2026 deployment: 3 seasons × 3 storage configurations.

Open in Spyder and press F5. Uses Gurobi, the 2026-10-17 continuous calibrated CVAE v3 / XGB models,
500->50 scenario reduction, alpha=1, beta=0.01 and original ERCOT
DA/RT settlement. No training or model selection occurs here.

IMPORTANT: runs 15 full *local market days* per case for the 7-day warm-up
and 7-day UTC scoring interval. The last local-market-day tail after the
UTC score ends is simulated solely to satisfy the original local-day runner.
It uses causal pre-context constants and is completely excluded from scoring.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, hashlib, time
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from holdout_runtime import SEASONS,load_season,install_adapters

# FROZEN, COMMON SETTINGS: edit only before FIRST scoring run.
SEASON_ORDER=('winter','spring','summer')
CASE_ORDER=('wind_only','ptes_50pct','ptes_100pct')
RAW_SCENARIOS=500
HL_SCENARIOS=50
TRACKING_ALPHA=1.0
SOC_BETA=0.01
HL_RISK_SCALE=1.0
WIND_RAMP_MW_PER_5MIN=40.0
HL_WIND_CURTAILMENT=True
SOC_HANDOVER='rollout'
LL_WARM_START=True
HL_BACKEND='gurobi'
LL_BACKEND='gurobi'
HL_TIME_LIMIT_S=300.0
HL_GAP=0.02
LL_TIME_LIMIT_S=120.0
LL_GAP=0.025
LL_MAX_ACCEPTABLE_TIME_LIMIT_GAP=0.05
FORECAST_MODEL='tcvae'
INITIAL_SOC_FRACTION=0.0
PROGRESS_EVERY=288  # One line per local day
MAKE_PER_CASE_PLOTS=False
DETAILED_LL_DIAGNOSTICS=False  # Set True before first run for large horizon archives
RESULTS_DIR=HERE/'results'/'HOLDOUT_2026_3SEASONS_3SYSTEMS_alpha1_CONTINUOUS_V3_ORIGINAL_REDUCTION'


def settings():
    return dict(seasons=SEASON_ORDER,cases=CASE_ORDER,raw_scenarios=RAW_SCENARIOS,
        hl_scenarios=HL_SCENARIOS,tracking_alpha=TRACKING_ALPHA,beta=SOC_BETA,
        hl_risk_scale=HL_RISK_SCALE,wind_ramp_mw=WIND_RAMP_MW_PER_5MIN,
        hl_wind_curtailment=HL_WIND_CURTAILMENT,soc_handover=SOC_HANDOVER,
        ll_warm_start=LL_WARM_START,hl_backend=HL_BACKEND,ll_backend=LL_BACKEND,
        hl_time_limit_s=HL_TIME_LIMIT_S,hl_gap=HL_GAP,
        ll_time_limit_s=LL_TIME_LIMIT_S,ll_gap=LL_GAP,
        ll_maximum_accepted_time_limit_gap=LL_MAX_ACCEPTABLE_TIME_LIMIT_GAP,
        forecast_model=FORECAST_MODEL,tcvae_checkpoint='continuous_cvae_v3_checkpoint.pt',tcvae_calibration='continuous_calibration_residual_paths.npz',
        initial_soc_fraction=INITIAL_SOC_FRACTION,
        make_per_case_plots=MAKE_PER_CASE_PLOTS,detailed_diagnostics=DETAILED_LL_DIAGNOSTICS,
        calendar='DA D-1 09:30 America/Chicago; scored interval UTC',
        semantics='50% PTES tank-only capacity reduction; same charge/discharge MW ratings')


def _source_hashes():
    names=['RUN_ALL_HOLDOUTS_FROM_SPYDER.py','holdout_runtime.py',
           'run_multiday.py','run_development_day.py','soc_rollout.py',
           'ptes_mpc/physics.py','ptes_mpc/hl_market.py','ptes_mpc/ll_mpc.py',
           'ptes_mpc/harness.py','ptes_mpc/linear_mip.py',
           'wind_only/runner.py','wind_only/hl.py','wind_only/ll.py',
           'forecast_release/source_releases/horse_creek_forecasting_release/models/continuous_cvae_v3_checkpoint.pt',
           'forecast_release/source_releases/horse_creek_forecasting_release/models/continuous_calibration_residual_paths.npz',
           'forecast_release/source_releases/horse_creek_forecasting_release/continuous_cvae.py',
           'forecast_release/source_releases/horse_creek_forecasting_release/continuous_cvae_provider.py',
           'forecast_release/source_releases/horse_creek_forecasting_release/train_thesis_tcvae.py',
           'forecast_release/source_releases/horse_creek_forecasting_release/horse_creek_forecasts.py',
           'forecast_release/source_releases/horse_creek_forecasting_release/horse_creek_copula.py',
           'forecast_release/source_releases/horse_creek_forecasting_release/original_backward_reduction.py',
           'forecast_release/ll_wind_xgb/direct_xgb_wind.json',
           'forecast_release/ll_rt_xgb/xgb_zero_delay_2025_trainJanJun.json']
    names += [f'holdout_inputs/{season}/{name}'
              for season in SEASON_ORDER for name in
              ('wind_5min.csv','DA_hourly.csv','RT15.csv','HL_realized_hourly_wind_DA.csv','LL_realized_15min_wind_RT.csv')]
    return {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}


def _fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()


def _market_dates(season):
    start,_,_=SEASONS[season]
    # Start Sunday local midnight (before the Monday UTC warmup begins),
    # preserving >7 full days of causal operating history before the score.
    # Run until the final Monday local midnight, after the UTC score closes.
    # Scoring uses ONLY the exact declared seven UTC days.
    return start,15


def _scored_financials(folder,active,case):
    efile=folder/('five_minute_wind_only.csv' if case=='wind_only' else 'five_minute_closed_loop.csv')
    qfile=folder/('da_commitments_wind_only.csv' if case=='wind_only' else 'da_commitments.csv')
    e=pd.read_csv(efile);q=pd.read_csv(qfile)
    e['time_utc']=pd.to_datetime(e['time_utc'],utc=True,format='mixed')
    q['hour_start_utc']=pd.to_datetime(q['hour_start_utc'],utc=True,format='mixed')
    a=active['score_start'];b=active['score_end']
    warm=e[e.time_utc<a]
    v=e[(e.time_utc>=a)&(e.time_utc<b)].copy().sort_values('time_utc')
    times=pd.date_range(a,b-pd.Timedelta(minutes=5),freq='5min',tz='UTC')
    if len(v)!=2016 or not v.time_utc.reset_index(drop=True).equals(pd.Series(times)):
        raise AssertionError(f'{case}: score must be exactly 2016 consecutive 5-min UTC records')
    if len(warm)<6*288:
        raise AssertionError('Insufficient warm-up observations')
    q=q.set_index('hour_start_utc')['DA_commitment_MW'].sort_index()
    hindex=pd.date_range(a,b-pd.Timedelta(hours=1),freq='h',tz='UTC')
    qq=q.reindex(hindex)
    da=active['hourly']['DA_price_usd_mwh'].reindex(hindex)
    if qq.isna().any() or da.isna().any():raise AssertionError('Missing scored DA bids or clearing prices')
    q15=qq.reindex(v.time_utc.dt.floor('h')).to_numpy(dtype=float)
    v['DA_commitment_MW']=q15
    v['DA_USD_per_MWh']=active['hourly']['DA_price_usd_mwh'].reindex(v.time_utc.dt.floor('h')).to_numpy(float)
    # Per-five-minute exact algebra equivalent to 15-minute signed settlement.
    v['RT_price_USD_per_MWh']=active['rt'].reindex(v.time_utc.dt.floor('15min')).to_numpy(float)
    if v.RT_price_USD_per_MWh.isna().any():raise AssertionError('Missing scored RT settlement price')
    v['DA_cash_USD']=v.DA_commitment_MW*v.DA_USD_per_MWh/12.
    v['RT_cash_USD']=(v.physical_poi_mw-v.DA_commitment_MW)*v.RT_price_USD_per_MWh/12.
    v['electricity_cash_USD']=v.DA_cash_USD+v.RT_cash_USD
    v['wind_curtailment_mwh']=v.curtailed_wind_mw/12
    v['net_poi_mwh']=v.physical_poi_mw/12
    v['utc_day']=v.time_utc.dt.strftime('%Y-%m-%d')
    energy_qtr=v.groupby(v.time_utc.dt.floor('15min')).apply(
        lambda block:float((block.physical_poi_mw-block.DA_commitment_MW).sum()/12.),include_groups=False)
    if len(energy_qtr)!=672:raise AssertionError('Expected exactly 672 settlement quarters')
    v['RT_energy_dev_mwh']=(v.physical_poi_mw-v.DA_commitment_MW)/12.
    values=dict(
        season=active['season'],case=case,storage_fraction={'wind_only':0,'ptes_50pct':.5,'ptes_100pct':1.}[case],
        score_start_utc=a.isoformat(),score_end_utc=b.isoformat(),
        scored_5min_steps=2016,scored_hours=168,scored_quarters=672,
        da_revenue_usd=float(v.DA_cash_USD.sum()),rt_revenue_usd=float(v.RT_cash_USD.sum()),
        total_electricity_revenue_usd=float(v.electricity_cash_USD.sum()),
        da_sold_mwh=float(qq.sum()),available_wind_mwh=float((v.available_wind_mw if case=='wind_only' else v.actual_wind_t0_mw).sum()/12),
        wind_delivered_mwh=float(v.dispatched_wind_mw.sum()/12),
        wind_curtailed_mwh=float(v.curtailed_wind_mw.sum()/12),
        net_poi_mwh=float(v.physical_poi_mw.sum()/12),
        abs_quarter_DA_deviation_mwh=float(energy_qtr.abs().sum()),
        final_soc_mwhth=(None if case=='wind_only' else float(v.tank_mwhth.iloc[-1])),
        initial_scored_soc_mwhth=(None if case=='wind_only' else float(warm.tank_mwhth.iloc[-1])),
        ptes_charge_mwh=(0. if case=='wind_only' else float(v.charge_mw.sum()/12)),
        ptes_discharge_mwh=(0. if case=='wind_only' else float(v.discharge_mw.sum()/12)),
        safety_interventions=(0 if case=='wind_only' else int(v.safety_override.sum())),
        wind_up_ramp_exceedances=int((v.dispatched_wind_mw.diff()>WIND_RAMP_MW_PER_5MIN+1e-4).sum()),
        ll_solver_failures=0,
        filled_wind_timestamps=active['wind_repairs'],filled_rt_timestamps=active['rt_repairs'],
        post_score_simulated_tail_excluded=True)
    if abs(values['da_revenue_usd']-float((qq*da).sum()))>1e-5:raise AssertionError('DA revenue mismatch')
    if abs(values['da_revenue_usd']+values['rt_revenue_usd']-values['total_electricity_revenue_usd'])>1e-4:
        raise AssertionError('DA/RT revenue does not reconcile')
    v.to_csv(folder/'SCORED_UTC_5MIN.csv',index=False)
    pd.DataFrame({'quarter_start_utc':energy_qtr.index,'energy_deviation_mwh':energy_qtr.values,
                  'rt_settlement_price_usd_mwh':active['rt'].reindex(energy_qtr.index).to_numpy(float)}).to_csv(
                  folder/'SCORED_UTC_QUARTERS.csv',index=False)
    daily=[]
    for day,g in v.groupby('utc_day'):
        d=dict(season=active['season'],case=case,utc_day=day,
             DA_USD=float(g.DA_cash_USD.sum()),RT_USD=float(g.RT_cash_USD.sum()),
             revenue_USD=float(g.electricity_cash_USD.sum()),
             wind_available_mwh=float((g.available_wind_mw if case=='wind_only' else g.actual_wind_t0_mw).sum()/12),
             wind_curtailed_mwh=float(g.curtailed_wind_mw.sum()/12),
             POI_mwh=float(g.physical_poi_mw.sum()/12),
             end_soc_mwhth=(None if case=='wind_only' else float(g.tank_mwhth.iloc[-1])))
        daily.append(d)
    if len(daily)!=7 or abs(sum(x['revenue_USD'] for x in daily)-values['total_electricity_revenue_usd'])>1e-5:
        raise AssertionError('UTC daily metrics must reconcile to week')
    pd.DataFrame(daily).to_csv(folder/'SCORED_UTC_DAILY.csv',index=False)
    (folder/'SCORED_UTC_SUMMARY.json').write_text(json.dumps(values,indent=2,default=str))
    return values,daily


def _draw_comparison(rows,daily,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=pd.DataFrame(rows)
    order=list(CASE_ORDER)
    fig,axs=plt.subplots(1,3,figsize=(16,5))
    for ax,season in zip(axs,SEASON_ORDER):
        subset=frame[frame.season==season].set_index('case').reindex(order)
        ax.bar(['No PTES','50% PTES','100% PTES'],subset.total_electricity_revenue_usd)
        ax.set_title(season.capitalize());ax.tick_params(axis='x',rotation=20)
        ax.set_ylabel('Seven-day DA + RT electricity revenue (USD)')
    fig.suptitle('Protected 2026 holdout comparison: same UTC week, independently optimized DA bids')
    fig.tight_layout();fig.savefig(out/'01_holdout_electricity_revenue.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,col,ylabel in [(axs[0],'wind_curtailed_mwh','Curtailment (MWh)'),
                          (axs[1],'abs_quarter_DA_deviation_mwh','Absolute quarterly DA deviation (MWh)')]:
        x=np.arange(3)
        for i,case in enumerate(order):
            d=frame[frame.case==case].set_index('season').reindex(SEASON_ORDER)
            ax.bar(x+(i-1)*.25,d[col],width=.25,label=case)
        ax.set_xticks(x,[s.capitalize() for s in SEASON_ORDER]);ax.set(ylabel=ylabel);ax.legend()
    fig.tight_layout();fig.savefig(out/'02_curtailment_and_DA_deviation.png',dpi=170);plt.close(fig)


def run_all():
    from VERIFY_HOLDOUT_PACKAGE_FROM_SPYDER import preflight
    preflight(check_gurobi=(HL_BACKEND=='gurobi' or LL_BACKEND=='gurobi'))
    import run_multiday
    from ptes_mpc.physics import PlantParams
    import wind_only.runner
    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    config=settings();sources=_source_hashes();config_hash=_fingerprint(dict(settings=config,source_hashes=sources))
    manifest=RESULTS_DIR/'FROZEN_EXPERIMENT_PROTOCOL.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['config_hash']!=config_hash:
            raise RuntimeError('FROZEN CONFIG MISMATCH: use a NEW output folder. Never retune a scored run in place.')
    else:
        manifest.write_text(json.dumps({'config_hash':config_hash,'settings':config,
                  'status':'frozen before performing any scored computation',
                  'source_sha256':sources,
                  'source_paper':'PTES_Wind_MPC__, Table 4, capacity sensitivity 25/50/75/100%',
                  'half_duration_interpretation':'half 2685.63 MWhth tank; unchanged MW electrical ratings',
                  'no_hyperparameter_tuning_on_holdouts':True},indent=2))
    all_rows=[];all_daily=[]
    for season in SEASON_ORDER:
        active=load_season(season)
        pt,wo=install_adapters(active)
        first,days=_market_dates(season)
        for case in CASE_ORDER:
            dest=RESULTS_DIR/season/case
            dest.mkdir(parents=True,exist_ok=True)
            (dest/'case_config.json').write_text(json.dumps({**config,'season':season,'case':case,
                      'score_start':active['score_start'].isoformat(),
                      'score_end':active['score_end'].isoformat(),
                      'data_fixes':dict(wind=active['wind_repairs'],rt=active['rt_repairs']),
                      'run_config_hash':config_hash},indent=2,default=str))
            summary_path=dest/'SCORED_UTC_SUMMARY.json'
            print(f'\n=== {season.upper()} / {case.upper()} ===',flush=True)
            if summary_path.exists():
                row=json.loads(summary_path.read_text())
                if (row.get('score_start_utc')!=active['score_start'].isoformat() or
                    row.get('score_end_utc')!=active['score_end'].isoformat()):
                    raise RuntimeError('Existing score interval inconsistent with declared holdout')
                print('Complete result detected; resuming without recomputation',flush=True)
                daily=pd.read_csv(dest/'SCORED_UTC_DAILY.csv').to_dict('records')
            else:
                if case=='wind_only':
                    wo.run(day=first,days=days,ns=HL_SCENARIOS,model=FORECAST_MODEL,
                        backend=LL_BACKEND,hl_backend=HL_BACKEND,tracking_weight=TRACKING_ALPHA,
                        hl_risk_scale=HL_RISK_SCALE,offer_cap_fraction=1.,
                        ll_time_limit=LL_TIME_LIMIT_S,ll_gap=LL_GAP,
                        hl_time_limit=HL_TIME_LIMIT_S,hl_gap=HL_GAP,
                        progress_every=PROGRESS_EVERY,out=dest,make_plots=MAKE_PER_CASE_PLOTS,
                        forecast_raw_scenarios=RAW_SCENARIOS,
                        wind_ramp_mw_per_5min=WIND_RAMP_MW_PER_5MIN)
                else:
                    full=PlantParams()
                    if case=='ptes_50pct':
                        from dataclasses import replace
                        prm=replace(full,capacity_mwhth=full.capacity_mwhth*.5)
                    else: prm=full
                    pt.run(day=first,days=days,ns=HL_SCENARIOS,model=FORECAST_MODEL,
                        start_fraction=INITIAL_SOC_FRACTION,finish_fraction=None,
                        backend=LL_BACKEND,hl_backend=HL_BACKEND,out=dest,
                        hl_time_limit=HL_TIME_LIMIT_S,ll_time_limit=LL_TIME_LIMIT_S,
                        beta=SOC_BETA,hl_gap=HL_GAP,ll_gap=LL_GAP,
                        tracking_policy='weighted',tracking_weight_usd_per_mwh=TRACKING_ALPHA,
                        tracking_budget_mwh=.25,curtailment_enabled=True,
                        gurobi_threads=0,gurobi_seed=1,gurobi_output=False,
                        progress_every=PROGRESS_EVERY,
                        ll_maximum_time_limit_gap=LL_MAX_ACCEPTABLE_TIME_LIMIT_GAP,
                        ll_warm_start_enabled=LL_WARM_START,make_plots=MAKE_PER_CASE_PLOTS,
                        display_plots=False,detailed_diagnostics=DETAILED_LL_DIAGNOSTICS,
                        ll_diagnostic_stride=3,hl_curtailment_enabled=HL_WIND_CURTAILMENT,
                        hl_risk_scale=HL_RISK_SCALE,plant_params=prm,
                        soc_handover=SOC_HANDOVER,rollout_policy='tracking_rule',
                        rollout_relax_hours=4.,rollout_anchor='hl_mean')
                row,daily=_scored_financials(dest,active,case)
            all_rows.append(row);all_daily.extend(daily)
            pd.DataFrame(all_rows).to_csv(RESULTS_DIR/'PARTIAL_COMPARISON.csv',index=False)
            print(f"SCORED {season} {case}: revenue=${row['total_electricity_revenue_usd']:,.2f}, "
                  f"curtail={row['wind_curtailed_mwh']:,.1f} MWh, "
                  f"abs DA dev={row['abs_quarter_DA_deviation_mwh']:,.1f} MWh",flush=True)
    frame=pd.DataFrame(all_rows)
    frame.to_csv(RESULTS_DIR/'ALL_9_CASES_COMPARISON.csv',index=False)
    pd.DataFrame(all_daily).to_csv(RESULTS_DIR/'ALL_63_UTC_DAYS.csv',index=False)
    gain=[]
    for season in SEASON_ORDER:
        f=frame[frame.season==season].set_index('case')
        for variant in ('ptes_50pct','ptes_100pct'):
            gain.append(dict(season=season,storage=variant,
                revenue_gain_vs_wind_only_usd=float(f.loc[variant,'total_electricity_revenue_usd']-
                                                    f.loc['wind_only','total_electricity_revenue_usd']),
                curtailment_change_mwh=float(f.loc[variant,'wind_curtailed_mwh']-
                                             f.loc['wind_only','wind_curtailed_mwh'])))
    pd.DataFrame(gain).to_csv(RESULTS_DIR/'PTES_PREMIUM_VS_WIND_ONLY.csv',index=False)
    _draw_comparison(all_rows,all_daily,RESULTS_DIR)
    (RESULTS_DIR/'COMPLETED_ALL_NINE.json').write_text(json.dumps(
        {'status':'completed','config_hash':config_hash,'seasons':list(SEASON_ORDER),
         'cases':list(CASE_ORDER),'total_cases':9},indent=2))
    print('\nALL 9 COMPLETE. UTC-scored revenue comparison:',flush=True)
    print(frame.pivot(index='season',columns='case',values='total_electricity_revenue_usd').to_string(),flush=True)
    print('Results:',RESULTS_DIR.resolve(),flush=True)
    return frame


if __name__=='__main__':
    run_all()
