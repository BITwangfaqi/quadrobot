"""Original prediction-driven loop, using the WBT model instead of B2/Z1."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from world_model import WebotsModel
from mpc_bridge import MPC, Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--gait', choices=('stand', 'trot', 'walk'), default='stand')
    parser.add_argument('--velocity', type=float, default=0.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('.cache/trajectory.npz'))
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('steps must be positive')
    settings = Settings(**(json.loads(args.config.read_text()) if args.config else {}))
    robot = WebotsModel()
    robot.set_gait_sequence(args.gait, settings.gait_period)
    mpc = MPC(robot, settings)
    q, v = robot.q0.copy(), np.zeros(robot.nv)
    history = {name: [] for name in ('q', 'v', 'a', 'tau', 'forces', 'cv', 'elapsed')}
    for step in range(args.steps):
        plan = mpc.solve(q, v, step*settings.dt_min, [args.velocity, 0, 0, 0, 0, 0])
        for name in history:
            value = getattr(plan, name)
            history[name].append(value[0] if isinstance(value, np.ndarray) else value)
        q, v = plan.q[1].copy(), plan.v[1].copy()
        print(f'{step+1}/{args.steps}: CV={plan.cv:.3g}, solve={plan.elapsed*1000:.1f}ms', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **history, dt=settings.dt_min, joint_names=robot.names)
    print('Saved', args.output)
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, sharex=True)
        t = np.arange(args.steps)*settings.dt_min
        for ax, name, offset in zip(axes, ('q', 'v', 'tau'), (7, 6, 0)):
            ax.plot(t, np.array(history[name])[:, offset:])
            ax.set_ylabel(name)
        axes[-1].set_xlabel('Model time (s)')
        axes[0].legend(robot.names, ncol=4, fontsize=6)
        plt.show()


if __name__ == '__main__':
    main()
