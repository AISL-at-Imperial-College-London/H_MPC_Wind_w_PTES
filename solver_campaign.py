#!/usr/bin/env python3
"""Select a solver for the unmodified original historical campaigns.

This is strictly an execution adapter. The main study scripts, solver modules,
forecast checkpoints, plant model and datasets are not patched on disk.
HiGHS has a different LL objective representation from Gurobi, so its
outputs go to a separate directory and MUST NOT be called Gurobi replications.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BASE_RESULTS = {
    'holdouts': 'HOLDOUT_2026_3SEASONS_3SYSTEMS_alpha1_CONTINUOUS_V3_ORIGINAL_REDUCTION',
    'deterministic': 'HOLDOUT_2026_3SEASONS_3SYSTEMS_alpha1_CONTINUOUS_V3_DETERMINISTIC_HL_WEIGHTED_MEAN',
    'extra': 'EXTRA_3WEEKS_V3_BOTH_POLICIES_18CASES',
}
CASE_COUNTS = {'holdouts': 9, 'deterministic': 9, 'extra': 18}


def results_dir(study, solver):
    # Gurobi folder names are those used by the original authors' runners.
    tail = '_HIGHS' if solver == 'highs' else ''
    return ROOT / 'results' / (BASE_RESULTS[study] + tail)


def run(study, solver='highs', dry_run=False):
    if study not in BASE_RESULTS or solver not in ('highs', 'gurobi'):
        raise ValueError('Unknown study or solver')
    folder = results_dir(study, solver)
    print(f'Campaign: {study} ({CASE_COUNTS[study]} cases)', flush=True)
    print(f'Solver: {solver}', flush=True)
    print(f'Output: {folder}', flush=True)
    if solver == 'highs':
        print('HiGHS is open source. PTES LL uses a piecewise-linear MILP '
              'approximation of the quadratic terminal objective.', flush=True)
        print('These are alternative-solver experiments, not identical '
              'replications of Gurobi MIQP results.', flush=True)
    if dry_run:
        print('Dry run only; no forecast or MPC solves have been executed.', flush=True)
        return 0

    import RUN_ALL_HOLDOUTS_FROM_SPYDER as primary
    primary.HL_BACKEND = solver
    primary.LL_BACKEND = solver
    if solver == 'highs':
        # Includes execution-adapter source in output protocol hashes, even
        # though the frozen scientific source hashes are left unchanged.
        orig_hashes = primary._source_hashes

        def backend_hashes():
            result = orig_hashes()
            for fname in ('run.py', 'solver_campaign.py'):
                result[fname] = hashlib.sha256((ROOT / fname).read_bytes()).hexdigest()
            return result

        primary._source_hashes = backend_hashes
    if study == 'holdouts':
        primary.RESULTS_DIR = folder
        primary.run_all()
    elif study == 'deterministic':
        import RUN_DETERMINISTIC_HOLDOUTS_FROM_SPYDER as deterministic
        deterministic.DETERMINISTIC_RESULTS_DIR = folder
        deterministic.run_all()
    elif study == 'extra':
        import RUN_V3_EXTRA_3_WEEKS_BOTH_FROM_SPYDER as extra
        extra.VERIFY_GUROBI = solver == 'gurobi'
        extra.RESULTS_DIR = folder
        extra.run_all()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description='Execute an unmodified campaign with selected solver')
    p.add_argument('study', choices=tuple(BASE_RESULTS))
    p.add_argument('--solver', choices=('highs', 'gurobi'), default='highs')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    return run(args.study, args.solver, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
