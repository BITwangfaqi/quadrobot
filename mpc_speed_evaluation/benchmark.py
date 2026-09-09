"""Sequential same-state replay benchmark; leaves controller code/logs intact."""
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('MPC_BENCHMARK_OUTPUT', Path(__file__).resolve().parent)).resolve()


def worker(kind, matched):
    import numpy as np
    sys.path.insert(0, str(ROOT / f'webots_{kind}_dynamics_mpc' / 'controllers' / f'{kind}_dynamics_mpc'))
    import mpc
    if matched and kind == 'inverse':
        mpc.R_ACC_CART = mpc.R_ACC_POLE = 0.0
    cls = getattr(mpc, f'{kind.title()}DynamicsMPC')
    start = perf_counter()
    controller = cls()
    build_ms = (perf_counter() - start) * 1000
    states = json.loads((OUT / 'states.json').read_text())
    records = []
    for index, state in enumerate(states):
        start = perf_counter()
        result = controller.solve(state)
        wall_ms = (perf_counter() - start) * 1000
        records.append(dict(index=index, solve_ms=result.solve_time_ms,
                            wall_ms=wall_ms, success=result.success,
                            status=result.status, force=result.force,
                            iterations=int(controller.opti.stats()['iter_count'])))
    print(json.dumps(dict(build_ms=build_ms, nx=int(controller.opti.nx),
                         ng=int(controller.opti.ng), records=records)))


def summary(records, key):
    import numpy as np
    a = np.array([float(r[key]) for r in records])
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                max=float(a.max()), over_20ms=int((a > 20).sum()))


def main():
    import casadi
    OUT.mkdir(parents=True, exist_ok=True)
    # Both methods receive exactly the same 751 states, at 20 ms intervals.
    source = ROOT / 'webots_forward_dynamics_mpc/logs/offline_validation.csv'
    with source.open() as f:
        rows = list(csv.DictReader(f))[::10]
    states = [[float(r[k]) for k in ('x', 'theta', 'x_dot', 'theta_dot')] for r in rows]
    (OUT / 'states.json').write_text(json.dumps(states))
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=sys.executable, version=sys.version,
                  platform=platform.platform(), casadi=casadi.__version__,
                  threads=1), replay_source=str(source), states=len(states), runs=[])
    # Alternate order, use fresh processes, and never compete for CPU in parallel.
    for repeat in range(3):
        order = ['forward', 'inverse'] if repeat % 2 == 0 else ['inverse', 'forward']
        for matched in (False, True):
            for kind in order:
                proc = subprocess.run([sys.executable, __file__, kind, str(int(matched))],
                                      env=env, text=True, capture_output=True, check=True)
                run = json.loads(proc.stdout)
                run.update(kind=kind, matched=matched, repeat=repeat)
                result['runs'].append(run)
                print(kind, 'matched' if matched else 'original', repeat,
                      summary(run['records'][1:], 'solve_ms'), flush=True)
    (OUT / 'raw_results.json').write_text(json.dumps(result, indent=2))
    aggregate = {}
    for matched in (False, True):
        for kind in ('forward', 'inverse'):
            runs = [r for r in result['runs'] if r['kind'] == kind and r['matched'] == matched]
            warm = [x for r in runs for x in r['records'][1:]]
            aggregate[f'{kind}_{"matched" if matched else "original"}'] = dict(
                solve=summary(warm, 'solve_ms'), wall=summary(warm, 'wall_ms'),
                iterations=summary(warm, 'iterations'),
                failures=sum(not x['success'] for r in runs for x in r['records']),
                first_solve_ms=[r['records'][0]['solve_ms'] for r in runs],
                nx=runs[0]['nx'], ng=runs[0]['ng'])
    (OUT / 'summary.json').write_text(json.dumps(aggregate, indent=2))
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    if len(sys.argv) == 3:
        worker(sys.argv[1], bool(int(sys.argv[2])))
    else:
        main()
