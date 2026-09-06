"""Measure complete solve calls, convergence and 12.5 ms deadline misses.

The state follows the last prediction for this solver benchmark. Use
webots_check.py for independent physical closed-loop validation.
"""
import argparse
from datetime import datetime, timezone
from dataclasses import asdict, replace
import json
import hashlib
import os
from pathlib import Path
import platform
import time
import numpy as np
from config import MPCConfig, MPCReference
from gait import ContactSchedule
from mpc import InverseDynamicsMPC
from rigid_body_model import WholeBodyModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('stand', 'trot', 'tool'), default='stand')
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('samples must be >=2')
    cfg = MPCConfig.from_json(args.config)
    model = WholeBodyModel()
    solver = InverseDynamicsMPC(model, cfg)
    q, v = model.neutral(), np.zeros(24)
    q[:3] = model.description.initial_position
    reference = MPCReference(q[:3].copy(), q[3:7].copy(), np.zeros(3), 0., q[7:].copy(),
                             model.tool_position(q), np.zeros(3), np.zeros(3))
    records = []
    for index in range(args.samples):
        elapsed = index * cfg.solve_period
        ref = reference
        if args.mode == 'trot':
            velocity = np.array([.06, 0., 0.])
            ref = replace(reference, base_position=reference.base_position + elapsed * velocity,
                          base_velocity_world=velocity, tool_position=reference.tool_position + elapsed * velocity,
                          tool_velocity=velocity)
        elif args.mode == 'tool':
            offset = .01 * np.sin(2 * np.pi * elapsed)
            ref = replace(reference, tool_position=reference.tool_position + [0., 0., offset])
        schedule = ContactSchedule('trot' if args.mode == 'trot' else 'stand', elapsed, cfg.gait_period)
        started = time.perf_counter()
        try:
            solution = solver.solve(q, v, ref, schedule)
            wall = time.perf_counter() - started
            records.append(dict(index=index, wall_ms=wall*1000, solver_ms=solution.solve_time*1000,
                                iterations=solution.iterations, violation=solution.constraint_violation,
                                success=True))
            # Advance to the actual 12.5 ms control period on the nonuniform grid.
            k = int(np.searchsorted(solver.times, cfg.solve_period, side='right') - 1)
            local = cfg.solve_period - solver.times[k]
            tangent = model.difference(solution.q[0], solution.q[k]) + local * solution.v[k]
            q = model.integrate(solution.q[0], tangent)
            v = solution.v[k] + local * solution.a[k]
        except RuntimeError as error:
            records.append(dict(index=index, wall_ms=(time.perf_counter()-started)*1000,
                                success=False, error=str(error)))
        if index % 10 == 0:
            print(records[-1], flush=True)
    steady = records[1:]
    times = np.array([r['wall_ms'] for r in steady])
    summary = dict(mean_ms=float(times.mean()), p95_ms=float(np.percentile(times, 95)),
                   p99_ms=float(np.percentile(times, 99)), max_ms=float(times.max()),
                   failures=sum(not r['success'] for r in records),
                   deadline_misses=sum(r['wall_ms'] > 12.5 for r in steady),
                   steady_samples=len(steady), cold=records[0])
    summary['mean_hz'] = 1000. / summary['mean_ms']
    summary['all_deadline_misses'] = sum(r['wall_ms'] > 12.5 for r in records)
    summary['meets_80hz_after_cold_start'] = summary['failures'] == 0 and summary['deadline_misses'] == 0
    summary['meets_80hz_every_sample'] = summary['failures'] == 0 and summary['all_deadline_misses'] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                if line.startswith('model name')), platform.processor())
    args.output.write_text(json.dumps(dict(config=asdict(cfg), mode=args.mode,
        recorded_at=datetime.now(timezone.utc).isoformat(), cpu=cpu, source_sha256=source_hashes,
        machine=platform.machine(), python=os.sys.executable, backend=model.casadi_backend,
        build_seconds=solver.build_time, cache_hit=solver.cache_hit, summary=summary, samples=records), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
