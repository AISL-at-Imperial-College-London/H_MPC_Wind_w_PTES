#!/usr/bin/env python3
"""Public command line interface for Horse Creek PTES experiments.

The published-case configuration uses Gurobi by default. Pass
``--solver highs`` to use the open-source MILP alternative; that option
runs the original experiment code in an isolated result directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RUNNERS = {
    'holdouts': 'RUN_ALL_HOLDOUTS_FROM_SPYDER.py',
    'deterministic': 'RUN_DETERMINISTIC_HOLDOUTS_FROM_SPYDER.py',
    'extra': 'RUN_V3_EXTRA_3_WEEKS_BOTH_FROM_SPYDER.py',
    'compare': 'COMPARE_V3_DETERMINISTIC_TO_STOCHASTIC.py',
}
CAMPAIGNS = ('holdouts', 'deterministic', 'extra')


def main(argv=None):
    p = argparse.ArgumentParser(description='Horse Creek wind–PTES historical MPC replay')
    p.add_argument('study', choices=('preflight', 'solver-check', *RUNNERS, 'tests'),
                   help='Preflight, small solver check, complete campaign, comparison, or unit tests')
    p.add_argument('--solver', choices=('gurobi', 'highs'), default='gurobi',
                   help='Optimisation backend for campaigns (default: gurobi for original study)')
    p.add_argument('--no-gurobi', action='store_true',
                   help='Legacy alias for disabling Gurobi validation on preflight only; '
                        'prefer --solver highs')
    p.add_argument('--dry-run', action='store_true',
                   help='Describe a campaign and output directory without running it')
    a = p.parse_args(argv)

    if a.no_gurobi and a.study != 'preflight':
        p.error('--no-gurobi is a preflight-only legacy option; use --solver highs for simulations')
    if a.dry_run and a.study not in CAMPAIGNS:
        p.error('--dry-run is available only for holdouts, deterministic and extra')
    if a.study == 'compare' and a.solver == 'highs':
        p.error('The archived comparison uses frozen Gurobi reference data; '
                'do not compare against it as a HiGHS-only result. '
                'Inspect the separate HiGHS output folders instead.')

    if a.study == 'tests':
        cmd = [sys.executable, '-m', 'pytest', '-q', 'tests']
        return subprocess.call(cmd, cwd=ROOT)
    if a.study == 'solver-check':
        print('Checking open-source HiGHS HL and LL optimisers (no Gurobi needed)...', flush=True)
        for cmd in (
            [sys.executable, 'TEST_DETERMINISTIC_BASELINE_FROM_SPYDER.py'],
            [sys.executable, '-m', 'pytest', '-q', 'tests/test_core.py::test_solve_ll_smoke'],
        ):
            code = subprocess.call(cmd, cwd=ROOT)
            if code:
                return code
        print('HiGHS HL/LL solver checks passed (not a full historical replay).', flush=True)
        return 0
    if a.study == 'preflight':
        check_gurobi = a.solver == 'gurobi' and not a.no_gurobi
        # Separate processes: extra_v3_runtime mutates season registry on import.
        for module in ('VERIFY_HOLDOUT_PACKAGE_FROM_SPYDER', 'VERIFY_V3_EXTRA_WEEKS_FROM_SPYDER'):
            code = f'from {module} import preflight; preflight(check_gurobi={check_gurobi})'
            status = subprocess.call([sys.executable, '-c', code], cwd=ROOT)
            if status:
                return status
        return 0

    if a.study in CAMPAIGNS and (a.solver == 'highs' or a.dry_run):
        cmd = [sys.executable, str(ROOT / 'solver_campaign.py'), a.study,
               '--solver', a.solver]
        if a.dry_run:
            cmd.append('--dry-run')
    else:
        cmd = [sys.executable, str(ROOT / RUNNERS[a.study])]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
