"""Compare identical MPC inputs/warm starts with interpreted and generated C functions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import casadi as ca
import numpy as np

from mpc_bridge import MPC, Settings
from world_model import WebotsModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path('.cache/deployed/solver_function.so'))
    parser.add_argument('--trace', type=Path, required=True, help='Webots telemetry JSONL with states and solves')
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--output', type=Path, default=Path('validation/compiled-benchmark.json'))
    args = parser.parse_args()
    library = args.library.resolve()
    metadata = json.loads(library.with_suffix('.json').read_text())
    robot = WebotsModel()
    robot.set_nominal(np.asarray(metadata['nominal']))
    settings = Settings.from_metadata(metadata['settings'])
    mpc = MPC(robot, settings)
    ocp = mpc.ocp
    native = (ocp.solver_function, mpc.decoder, ocp.g_data)
    # Exercise the same deployment checks as the live controller.
    mpc._load_compiled(str(library))
    compiled = (ocp.solver_function, mpc.decoder, ocp.g_data)
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    states = [row for row in rows if row['event'] == 'state']
    solve_times = [row['time'] for row in rows if row['event'] == 'solve'][:args.samples]
    results = []
    gait, gait_start = 'stand', solve_times[0]
    for index, stamp in enumerate(solve_times):
        state = min(states, key=lambda row: abs(row['time']-stamp))
        if state['gait'] != gait:
            gait, gait_start = state['gait'], stamp
            robot.gait_sequence.gait_type = gait
            fresh = type(robot.gait_sequence)(gait, settings.gait_period)
            robot.gait_sequence.__dict__.update(fresh.__dict__)
        ocp.set_tracking_targets(np.zeros(6), np.zeros(3), np.asarray(settings.arm_force))
        ocp.update_params(np.r_[state['q'], state['v']], stamp-gait_start+1e-9)
        inputs = ocp.get_solver_params()
        params = ocp.opti.value(ocp.opti.p)
        outputs, timings, cvs = {}, {}, {}
        # Alternate order to avoid systematically giving one backend a warm CPU.
        order = [('interpreted', native), ('compiled', compiled)]
        if index % 2:
            order.reverse()
        for label, (solver, decoder, constraints) in order:
            started = time.perf_counter()
            solution = solver(*inputs)
            solver_seconds = time.perf_counter()-started
            decoded = decoder(solution, params)
            g, lo, hi = constraints(solution, params)
            cvs[label] = float(ocp.constr_viol_norm_inf(g, lo, hi))
            timings[label] = {'solver_seconds': solver_seconds,
                              'solver_decode_constraints_seconds': time.perf_counter()-started}
            outputs[label] = solution
        difference = float(np.max(np.abs(np.asarray(outputs['compiled']-outputs['interpreted']))))
        close = bool(np.allclose(outputs['compiled'], outputs['interpreted'], atol=1e-7, rtol=1e-7))
        # Decoding/constraint evaluation on identical solutions must also agree.
        for nf, cf in zip(native[1:], compiled[1:]):
            for left, right in zip(nf(outputs['interpreted'], params), cf(outputs['interpreted'], params)):
                np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-7)
        solution = outputs['interpreted']
        g, lo, hi = native[2](solution, params)
        accepted = (np.isfinite(np.asarray(solution)).all()
                    and float(ocp.constr_viol_norm_inf(g, lo, hi)) <= settings.max_cv)
        if accepted:
            ocp.retract_stacked_sol(solution, retract_all=False)
        else:
            mpc.reset()
        results.append({'time': stamp, 'state_time': state['time'], 'gait': gait,
                        'timings': timings, 'cv': cvs, 'max_solution_difference': difference,
                        'allclose': close, 'warm_start_reset': not bool(accepted)})
        print(f'{index+1}/{len(solve_times)} {gait}: native={timings["interpreted"]["solver_seconds"]:.4f}s '
              f'C={timings["compiled"]["solver_seconds"]:.4f}s close={close}', flush=True)
    summary = {}
    for label in ('interpreted', 'compiled'):
        values = [item['timings'][label]['solver_seconds'] for item in results]
        summary[label] = {'mean_seconds': float(np.mean(values)),
                          'median_seconds': float(np.median(values)),
                          'p95_seconds': float(np.percentile(values, 95))}
    summary['mean_speedup'] = summary['interpreted']['mean_seconds']/summary['compiled']['mean_seconds']
    summary['all_solutions_close'] = all(item['allclose'] for item in results)
    summary['samples'] = len(results)
    report = {'summary': summary, 'library_sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
              'trace': str(args.trace.resolve()), 'trace_sha256': hashlib.sha256(args.trace.read_bytes()).hexdigest(),
              'scope': 'Same measured states, parameters and warm starts; Fatrop only, no fallback. Rejected solutions reset the warm start. Excludes Python updates and retraction.',
              'samples': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
