"""Paired initial/final SoC experiment setup, WITHOUT reading protected holdouts."""
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from .physics import PlantParams,PlantState
from .harness import simulate_5min

@dataclass(frozen=True)
class SoCCase:
    name: str
    start_fraction: float
    finish_fraction: float|None

SOC_CASES=(
    SoCCase('empty_to_empty',0.,0.),
    SoCCase('full_to_full',1.,1.),
    SoCCase('empty_to_free',0.,None),
    SoCCase('full_to_free',1.,None),
)

def run_four_soc_cases(*,base_kwargs,prm:PlantParams=PlantParams(),
                       output_dir=None,terminal_beta_usd_per_mwhth2=1.,
                       initial_mode='off',initial_off_elapsed_steps=None,
                       hl_schedule_factory=None,allow_shared_hl_for_tests=False):
    """Run four completely separate trajectories with identical forecasts/prices.

    base_kwargs: kwargs for simulate_5min excluding initial_state,
    episode_terminal_target_mwhth, prm and terminal_beta_usd_per_mwhth2.
    Historical input data and current time restrictions remain caller responsibilities.
    The same *forecast inputs and true prices* are used across cases, but each case
    MUST regenerate its own HL commitments and reference from its initial SoC,
    by supplying hl_schedule_factory(case, initial_state) -> dict with keys
    'da_commitments_hourly' and 'soc_target_at'. Shared HL is allowed only for
    isolated unit tests by explicitly setting allow_shared_hl_for_tests=True.
    No free-terminal case receives an artificial cash credit for remaining inventory.
    """
    if hl_schedule_factory is None and not allow_shared_hl_for_tests:
        raise ValueError('Provide case-specific HL commitments/reference using hl_schedule_factory; '
                         'shared HL schedule is valid only for isolated LL testing')
    out={}
    for case in SOC_CASES:
        state=PlantState(case.start_fraction*prm.capacity_mwhth,initial_mode,
                prm.off_min_steps if initial_off_elapsed_steps is None else initial_off_elapsed_steps)
        final=(None if case.finish_fraction is None else case.finish_fraction*prm.capacity_mwhth)
        case_inputs=dict(base_kwargs)
        if hl_schedule_factory is not None:
            overrides=hl_schedule_factory(case,state)
            if not {'da_commitments_hourly','soc_target_at'} <= set(overrides):
                raise ValueError('HL factory must provide DA commitments and SoC reference for each case')
            case_inputs.update(overrides)
        out[case.name]=simulate_5min(**case_inputs,initial_state=state,
                episode_terminal_target_mwhth=final,
                terminal_beta_usd_per_mwhth2=terminal_beta_usd_per_mwhth2,prm=prm)
    if output_dir is not None:
        folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
        summary=[]
        for name,result in out.items():
            result.events.to_csv(folder/f'{name}_5min.csv',index=False)
            summary.append({'case':name,**result.cashflow,
                'initial_energy_mwhth':result.initial_energy_mwhth,
                'final_energy_mwhth':result.final_state.energy_mwhth,
                'final_mode':result.final_state.mode,
                'target_energy_mwhth':result.requested_final_energy_mwhth,
                'final_target_error_mwhth':result.actual_terminal_error_mwhth,
                'terminal_soft_penalty_usd_not_market_cashflow':result.terminal_soft_penalty_usd,
                'regulatory_interventions':result.regulatory_interventions,
                'forced_off_interventions':result.forced_off_interventions,
                'clipped_discharge_mwh':result.clipped_discharge_mwh,
                'max_ptes_step_change_mw':result.max_executed_ptes_power_ramp_mw_per_5min,
                'solve_failures':result.solve_failures})
        pd.DataFrame(summary).to_csv(folder/'case_summary.csv',index=False)
        (folder/'case_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    return out
