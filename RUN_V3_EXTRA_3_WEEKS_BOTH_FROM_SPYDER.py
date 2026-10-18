"""Single unattended V3 study: three extra UTC weeks x three systems x BOTH policies.

Run this ONE file (F5) from the extracted HC_MPC_V3 directory in Spyder.
Uses continuous calibrated CVAE V3 forecasting, 500 raw -> 50 retained,
and identical low-level controllers for two HL policies:
   stochastic = optimise original 50 joint, weighted scenarios;
   deterministic = optimise single probability-weighted joint mean;
The original 50 scenarios are still generated under deterministic HL.

Week definition, causal short warmup and UTC revenue scoring are carried
from the older Horse_Creek_3_Extra_Weeks package, NOT from the 2026 protected
holdout dates. Results are *supplemental historical/development studies*.

Resuming: rerun the same file. Completed SCORED_UTC_SUMMARY.json with complete
forecast audit skips the case; interrupted cases restart from initial state.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv, hashlib, json, sys, traceback
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))

import RUN_ALL_HOLDOUTS_FROM_SPYDER as v3
from extra_v3_runtime import (SEASONS,load_season,install_adapters,
                              expected_market_days,run_metadata)
from extra_v3_scoring import _scored_financials
from deterministic_hl_baseline import mean_hl_inputs, ensemble_sha256

# Study definition. All optimizer/forecast hyperparameters are inherited
# directly from the verified V3 stochastic 3x3 study, not tuned here.
SEASON_ORDER=('dec_2025','spring_dev','summer_dev')
CASE_ORDER=('wind_only','ptes_50pct','ptes_100pct')
POLICY_ORDER=('stochastic','deterministic')
RESULTS_DIR=HERE/'results'/'EXTRA_3WEEKS_V3_BOTH_POLICIES_18CASES'
CONTINUE_AFTER_CASE_ERROR=True # complete as much of the unattended night as possible
VERIFY_GUROBI=True


def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings():
    return dict(v3_settings=v3.settings(),months=SEASON_ORDER,
        systems=CASE_ORDER,policies=POLICY_ORDER,
        v3_stochastic_ensemble='500 raw -> 50 weighted, V3 original reduction',
        v3_deterministic='same 50 weighted -> ONE mean at HL solver boundary',
        score='7 exact UTC days, DA+RT only, no operating costs',
        warmup='original short-extra-week causal bootstrap, not standard 15-day holdout',
        risk_reference='pre-Dec-2025 historic data for December, pre-2026 for May/July',
        data_status='SUPPLEMENTARY historical/development/reused context; not independent holdouts',
        no_retuning=True,
        code='RUN_V3_EXTRA_3_WEEKS_BOTH_FROM_SPYDER.py')


def sources():
    result=v3._source_hashes()
    for name in ('RUN_V3_EXTRA_3_WEEKS_BOTH_FROM_SPYDER.py',
                 'extra_v3_runtime.py','extra_v3_scoring.py',
                 'deterministic_hl_baseline.py','VERIFY_V3_EXTRA_WEEKS_FROM_SPYDER.py'):
        result[name]=sha256(HERE/name)
    for season in SEASON_ORDER:
        for filename in ('wind_5min.csv','DA_hourly.csv','RT15.csv'):
            rel=f'holdout_inputs/{season}/{filename}'
            result[rel]=sha256(HERE/rel)
    return result


def fingerprint(config,hashes):
    return hashlib.sha256(json.dumps(dict(config=config,source_hashes=hashes),
                                     sort_keys=True).encode()).hexdigest()


@dataclass
class HLAudit:
    output:Path
    season:str
    case:str
    policy:str
    first_market_day:str
    expected_days:int
    count:int=0

    @staticmethod
    def _append(path,rows):
        if not rows:return
        exists=path.exists() and path.stat().st_size>0
        with path.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            if not exists:writer.writeheader()
            writer.writerows(rows)

    def record(self,w,p,weights):
        mean_w,mean_p,_=mean_hl_inputs(w,p,weights)
        self.count+=1
        if self.count>self.expected_days:
            raise AssertionError(f'{self.policy}/{self.season}/{self.case}: too many HL issues')
        day=(pd.Timestamp(self.first_market_day)+pd.Timedelta(days=self.count-1)).strftime('%Y-%m-%d')
        midnight=pd.Timestamp(day).tz_localize('America/Chicago').tz_convert('UTC')
        issue=midnight-pd.Timedelta(hours=14,minutes=30)
        audit=dict(hl_issue_index=self.count,delivery_day_local=day,
                   issue_utc=issue.isoformat(),case=self.case,policy=self.policy,
                   scenarios_passed_to_HL=(50 if self.policy=='stochastic' else 1),
                   retained_ensemble_scenarios=50,
                   original_ensemble_sha256=ensemble_sha256(w,p,weights),
                   retained_probability_sum=float(np.asarray(weights).sum()))
        self._append(self.output/'HL_ISSUE_AUDIT.csv',[audit])
        means=[dict(hl_issue_index=self.count,delivery_day_local=day,
                    hour_start_utc=(midnight+pd.Timedelta(hours=h)).isoformat(),
                    mean_wind_mw=float(mean_w[0,h]),
                    mean_DA_price_usd_mwh=float(mean_p[0,h])) for h in range(84)]
        self._append(self.output/'HL_WEIGHTED_MEANS.csv',means)
        # For stochastic only an audit is performed: send all original 50
        # scenarios into exactly the original V3 HL solver, unchanged.
        if self.policy=='stochastic':
            return w,p,weights
        # Deterministic: replace HL solver input only, NOT the generator or
        # state/SOC handover, with probability weighted 1-path expected value.
        return mean_w,mean_p,np.ones(1,dtype=float)

    def validate(self):
        if self.count!=self.expected_days:
            raise AssertionError(f'{self.policy}/{self.season}/{self.case}: expected {self.expected_days} HL issues, got {self.count}')
        for name,expected in [('HL_ISSUE_AUDIT.csv',self.expected_days),
                              ('HL_WEIGHTED_MEANS.csv',self.expected_days*84)]:
            rows=pd.read_csv(self.output/name)
            if len(rows)!=expected:
                raise AssertionError(f'{self.output/name}: {len(rows)} rows vs expected {expected}')
            if name=='HL_ISSUE_AUDIT.csv' and rows.hl_issue_index.tolist()!=list(range(1,expected+1)):
                raise AssertionError('Non-sequential or duplicated HL issues')


def _completed_case_ok(folder,active,days):
    scored=folder/'SCORED_UTC_SUMMARY.json'
    if not scored.is_file():return False
    row=json.loads(scored.read_text(encoding='utf-8'))
    if (row.get('score_start_utc')!=active['score_start'].isoformat() or
        row.get('score_end_utc')!=active['score_end'].isoformat()):
        raise RuntimeError(f'{folder}: score dates incompatible with current protocol')
    for name,expected in [('HL_ISSUE_AUDIT.csv',days),('HL_WEIGHTED_MEANS.csv',days*84)]:
        if not (folder/name).exists() or len(pd.read_csv(folder/name))!=expected:
            raise RuntimeError(f'{folder}: marked complete but HL forecast audit invalid: {name}')
    if not (folder/'SCORED_UTC_DAILY.csv').exists():
        raise RuntimeError(f'{folder}: marked complete but missing UTC daily results')
    return True


def _compare_pair(season,case):
    root=RESULTS_DIR
    a=root/'stochastic'/season/case
    b=root/'deterministic'/season/case
    if not ((a/'SCORED_UTC_SUMMARY.json').exists() and (b/'SCORED_UTC_SUMMARY.json').exists()):return None
    aa=pd.read_csv(a/'HL_ISSUE_AUDIT.csv')
    bb=pd.read_csv(b/'HL_ISSUE_AUDIT.csv')
    aw=pd.read_csv(a/'HL_WEIGHTED_MEANS.csv')
    bw=pd.read_csv(b/'HL_WEIGHTED_MEANS.csv')
    if len(aa)!=len(bb) or len(aw)!=len(bw):raise AssertionError('Mismatched number of forecasts')
    hashes_equal=(aa.original_ensemble_sha256.tolist()==bb.original_ensemble_sha256.tolist())
    max_w=float(np.abs(aw.mean_wind_mw-bw.mean_wind_mw).max())
    max_p=float(np.abs(aw.mean_DA_price_usd_mwh-bw.mean_DA_price_usd_mwh).max())
    valid=bool(hashes_equal and max_w<1e-6 and max_p<1e-6)
    s=json.loads((a/'SCORED_UTC_SUMMARY.json').read_text())
    d=json.loads((b/'SCORED_UTC_SUMMARY.json').read_text())
    row=dict(season=season,case=case,scenario_ensemble_sha256_exact_match=hashes_equal,
        forecast_means_match=valid,max_wind_mean_abs_difference_mw=max_w,
        max_DA_price_mean_abs_difference_usd_mwh=max_p,
        stochastic_revenue_usd=s['total_electricity_revenue_usd'],
        deterministic_revenue_usd=d['total_electricity_revenue_usd'],
        stochastic_DA_revenue_usd=s['da_revenue_usd'],
        deterministic_DA_revenue_usd=d['da_revenue_usd'],
        stochastic_RT_revenue_usd=s['rt_revenue_usd'],
        deterministic_RT_revenue_usd=d['rt_revenue_usd'],
        revenue_difference_stoch_minus_det_usd=s['total_electricity_revenue_usd']-d['total_electricity_revenue_usd'],
        stochastic_curtailment_mwh=s['wind_curtailed_mwh'],
        deterministic_curtailment_mwh=d['wind_curtailed_mwh'],
        stochastic_abs_DA_deviation_mwh=s['abs_quarter_DA_deviation_mwh'],
        deterministic_abs_DA_deviation_mwh=d['abs_quarter_DA_deviation_mwh'],
        stochastic_score_open_soc_mwhth=s['initial_scored_soc_mwhth'],
        deterministic_score_open_soc_mwhth=d['initial_scored_soc_mwhth'],
        stochastic_score_end_soc_mwhth=s['final_soc_mwhth'],
        deterministic_score_end_soc_mwhth=d['final_soc_mwhth'])
    if not valid:
        raise AssertionError(f'{season}/{case}: mismatch in V3 joint forecast inputs between policies (sha match={hashes_equal}, max wind={max_w}, max DA price={max_p})')
    return row


def _finish_partial():
    root=RESULTS_DIR
    rows=[];daily=[];paired=[]
    for season in SEASON_ORDER:
        for case in CASE_ORDER:
            for policy in POLICY_ORDER:
                folder=root/policy/season/case
                p=folder/'SCORED_UTC_SUMMARY.json'
                if p.exists():
                    rows.append(dict(hl_policy=policy,**json.loads(p.read_text())))
                    d=folder/'SCORED_UTC_DAILY.csv'
                    if d.exists():
                        df=pd.read_csv(d);df.insert(0,'hl_policy',policy)
                        daily.extend(df.to_dict('records'))
            try:
                pair=_compare_pair(season,case)
                if pair is not None:paired.append(pair)
            except Exception as exc:
                # The partially completed scored cases must remain inspectable.
                failure=root/'COMPARISON_FAILURES.log'
                with failure.open('a',encoding='utf-8') as f:f.write(f'{season}/{case}: {exc}\n')
                print('COMPARISON ERROR:',exc,flush=True)
    if rows:pd.DataFrame(rows).to_csv(root/'ALL_COMPLETED_CASES.csv',index=False)
    if daily:pd.DataFrame(daily).to_csv(root/'ALL_COMPLETED_UTC_DAYS.csv',index=False)
    if paired:pd.DataFrame(paired).to_csv(root/'MATCHED_STOCH_VS_DET.csv',index=False)
    print(f'Saved partial tables: {len(rows)}/18 cases, {len(paired)}/9 matched pairs.',flush=True)
    return rows,paired


def _solve_case(pt,wo,season,case,policy,active,days,output,runner):
    from ptes_mpc.physics import PlantParams
    first=SEASONS[season][0]
    if case=='wind_only':
        return wo.run(day=first,days=days,ns=v3.HL_SCENARIOS,model=v3.FORECAST_MODEL,
             backend=v3.LL_BACKEND,hl_backend=v3.HL_BACKEND,tracking_weight=v3.TRACKING_ALPHA,
             hl_risk_scale=v3.HL_RISK_SCALE,offer_cap_fraction=1.,
             ll_time_limit=v3.LL_TIME_LIMIT_S,ll_gap=v3.LL_GAP,
             hl_time_limit=v3.HL_TIME_LIMIT_S,hl_gap=v3.HL_GAP,
             progress_every=v3.PROGRESS_EVERY,out=output,make_plots=v3.MAKE_PER_CASE_PLOTS,
             forecast_raw_scenarios=v3.RAW_SCENARIOS,
             wind_ramp_mw_per_5min=v3.WIND_RAMP_MW_PER_5MIN)
    prm=PlantParams()
    if case=='ptes_50pct':
        from dataclasses import replace
        prm=replace(prm,capacity_mwhth=prm.capacity_mwhth*.5)
    return pt.run(day=first,days=days,ns=v3.HL_SCENARIOS,model=v3.FORECAST_MODEL,
        start_fraction=v3.INITIAL_SOC_FRACTION,finish_fraction=None,
        backend=v3.LL_BACKEND,hl_backend=v3.HL_BACKEND,out=output,
        hl_time_limit=v3.HL_TIME_LIMIT_S,ll_time_limit=v3.LL_TIME_LIMIT_S,
        beta=v3.SOC_BETA,hl_gap=v3.HL_GAP,ll_gap=v3.LL_GAP,
        tracking_policy='weighted',tracking_weight_usd_per_mwh=v3.TRACKING_ALPHA,
        tracking_budget_mwh=.25,curtailment_enabled=True,
        gurobi_threads=0,gurobi_seed=1,gurobi_output=False,
        progress_every=v3.PROGRESS_EVERY,
        ll_maximum_time_limit_gap=v3.LL_MAX_ACCEPTABLE_TIME_LIMIT_GAP,
        ll_warm_start_enabled=v3.LL_WARM_START,make_plots=v3.MAKE_PER_CASE_PLOTS,
        display_plots=False,detailed_diagnostics=v3.DETAILED_LL_DIAGNOSTICS,
        ll_diagnostic_stride=3,hl_curtailment_enabled=v3.HL_WIND_CURTAILMENT,
        hl_risk_scale=v3.HL_RISK_SCALE,plant_params=prm,
        soc_handover=v3.SOC_HANDOVER,rollout_policy='tracking_rule',
        rollout_relax_hours=4.,rollout_anchor='hl_mean')


def run_all():
    from VERIFY_V3_EXTRA_WEEKS_FROM_SPYDER import preflight
    preflight(check_gurobi=VERIFY_GUROBI)
    import run_multiday as pt
    import wind_only.runner as wo
    old_pt_solve=pt.solve_hl_market
    old_wo_solve=wo.solve_wind_hl
    if getattr(old_pt_solve,'_extra_v3_wrapper',False):
        raise RuntimeError('An earlier run is still installed in the same Python interpreter. Restart interpreter.')
    current={'audit':None}
    def ptes_hl(wind_scenarios,da_scenarios,weights,initial_energy_mwhth,
                rt_spread_hourly,risk_premium_hourly,**kwargs):
        a=current['audit']
        if a is None or a.case=='wind_only':raise RuntimeError('PTES HL solver outside matching audit')
        w,p,pi=a.record(wind_scenarios,da_scenarios,weights)
        return old_pt_solve(w,p,pi,initial_energy_mwhth,
                            rt_spread_hourly,risk_premium_hourly,**kwargs)
    def wind_hl(wind_scenarios,da_scenarios,weights,spread_hourly,risk_hourly,**kwargs):
        a=current['audit']
        if a is None or a.case!='wind_only':raise RuntimeError('Wind HL solver outside matching audit')
        w,p,pi=a.record(wind_scenarios,da_scenarios,weights)
        return old_wo_solve(w,p,pi,spread_hourly,risk_hourly,**kwargs)
    ptes_hl._extra_v3_wrapper=True
    wo.solve_wind_hl=wind_hl
    pt.solve_hl_market=ptes_hl
    try:
        config=settings(); hashes=sources(); protocol_hash=fingerprint(config,hashes)
        RESULTS_DIR.mkdir(parents=True,exist_ok=True)
        manifest=RESULTS_DIR/'FROZEN_EXPERIMENT_PROTOCOL.json'
        if manifest.exists():
            old=json.loads(manifest.read_text())
            if old['protocol_sha256']!=protocol_hash:
                raise RuntimeError(f'Config/source mismatch in {manifest}. Do not mix two experiments in one result folder.')
        else:
            manifest.write_text(json.dumps(dict(protocol_sha256=protocol_hash,settings=config,
                source_sha256=hashes,extra_week_metadata=run_metadata(),
                scientific_caveat='Historical replay, not unseen independent holdout',
                generated_before_scored_run=True),indent=2))
        print('\nStarting V3 3 weeks x 3 systems x 2 HL policies (18 cases)',flush=True)
        print('Results:',RESULTS_DIR.resolve(),flush=True)
        failures=[]
        for season in SEASON_ORDER:
            active=load_season(season)
            pt,wo=install_adapters(active)
            days=expected_market_days(season)
            for case in CASE_ORDER:
                for policy in POLICY_ORDER:
                    output=RESULTS_DIR/policy/season/case
                    output.mkdir(parents=True,exist_ok=True)
                    cfg=output/'case_config.json'
                    if not cfg.is_file():
                        cfg.write_text(json.dumps(dict(season=season,case=case,policy=policy,
                            score_start=active['score_start'].isoformat(),
                            score_end=active['score_end'].isoformat(),
                            protocol_sha256=protocol_hash,days=days,
                            wind_repairs=active['wind_repairs'],rt_repairs=active['rt_repairs'],
                            supplementary_metadata=run_metadata()[season]),indent=2,default=str))
                    elif json.loads(cfg.read_text())['protocol_sha256']!=protocol_hash:
                        raise RuntimeError(f'Case configuration mismatch: {cfg}')
                    print(f'\n=== {season} / {case} / {policy} ({days} local days) ===',flush=True)
                    if _completed_case_ok(output,active,days):
                        print('Complete result + forecast audit found; skipping.',flush=True)
                        _finish_partial()
                        continue
                    # Incomplete cases always restart from frozen initial
                    # episode conditions. Remove derived audit logs ONLY.
                    for name in ('HL_ISSUE_AUDIT.csv','HL_WEIGHTED_MEANS.csv',
                                 'SCORED_UTC_SUMMARY.json','SCORED_UTC_DAILY.csv'):
                        (output/name).unlink(missing_ok=True)
                    audit=HLAudit(output,season,case,policy,SEASONS[season][0],days)
                    current['audit']=audit
                    try:
                        _solve_case(pt,wo,season,case,policy,active,days,output,runner=None)
                        audit.validate()
                        row,_daily=_scored_financials(output,active,case)
                        (output/'FAILURE.txt').unlink(missing_ok=True)
                        print(f'COMPLETED {season}/{case}/{policy}: revenue ${row["total_electricity_revenue_usd"]:,.2f}, '
                              f'curtailment {row["wind_curtailed_mwh"]:,.1f} MWh',flush=True)
                    except Exception as exc:
                        msg=f'{season}/{case}/{policy}: {type(exc).__name__}: {exc}'
                        failures.append(msg)
                        (output/'FAILURE.txt').write_text(traceback.format_exc())
                        print('CASE FAILED:',msg,flush=True)
                        if not CONTINUE_AFTER_CASE_ERROR:raise
                    finally:
                        current['audit']=None
                    _finish_partial()
        rows,pairs=_finish_partial()
        if len(rows)==18 and len(pairs)==9 and not failures:
            (RESULTS_DIR/'COMPLETED_ALL_18_CASES.json').write_text(json.dumps(dict(
                 status='complete',total_cases=18,matched_pairs=9,
                 protocol_sha256=protocol_hash),indent=2))
            print('\nCOMPLETE: 18 cases, 9 matched stochastic-deterministic pairs.',flush=True)
        else:
            print('\nINCOMPLETE: check case-specific FAILURE.txt and COMPARISON_FAILURES.log',flush=True)
            (RESULTS_DIR/'INCOMPLETE_REPORT.json').write_text(json.dumps(dict(
                 cases_complete=len(rows),pairs_matched=len(pairs),
                 case_failures=failures),indent=2))
        if pairs:
            report=pd.DataFrame(pairs)
            print('\nMatched seven-day revenue difference (STOCH - DET) USD:')
            print(report.pivot(index='season',columns='case',values='revenue_difference_stoch_minus_det_usd').to_string())
        print('All outputs:',RESULTS_DIR.resolve(),flush=True)
        return rows,pairs
    finally:
        pt.solve_hl_market=old_pt_solve
        wo.solve_wind_hl=old_wo_solve

if __name__=='__main__':
    run_all()
