"""Run an isolated physical test using the exact WBT Robot on primitive flat ground.

Does not change the user's world or attach to their external controller instance.
Requires a display usable by Webots. All generated files stay in validation/runs.
"""
import argparse
import json
import inspect
import hashlib
from dataclasses import asdict
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def robot_text(text):
    start = re.search(r'^Robot\s*\{', text, re.M).start()
    depth = 0
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}]', text[start:]):
        token = match.group()
        if token == '{':
            depth += 1
        elif token == '}':
            depth -= 1
            if depth == 0:
                return text[start:start+match.end()]
    raise ValueError('Unbalanced Robot node')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--scenario', choices=('manual', 'trot', 'arm'), default='manual')
    parser.add_argument('--velocity', type=float, default=0.)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--analyze', type=Path, help='Analyze an existing run; supply its original scenario/config/duration/velocity')
    parser.add_argument('--timeout', type=float, default=600.)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    world = root.parents[1]/'worlds/quadruped_arm.wbt'
    if args.analyze:
        run = args.analyze.resolve()
    else:
        run = root/'validation/runs'/f'{args.scenario}-{time.time_ns()}'
        controllers = run/'controllers/probe'; controllers.mkdir(parents=True)
        worlds = run/'worlds'; worlds.mkdir()
        model_text = robot_text(world.read_text())
        model_text = model_text.replace('controller "<extern>"', 'controller "probe"\n  supervisor TRUE')
        # Resolve meshes relative to the original world, preserving all Physics,
        # JointParameters, Motors, sensors and collision geometries verbatim.
        def mesh_path(match):
            value = match.group(1)
            return json.dumps(str((world.parent/value).resolve())) if value.startswith('../') else match.group()
        model_text = re.sub(r'"([^"\n]+\.(?:STL|stl|obj))"', mesh_path, model_text)
        header = '''#VRML_SIM R2023a utf8
    WorldInfo { basicTimeStep 10 }
    Viewpoint { position 1 1 1 orientation -0.4 0.8 0.4 2.3 }
    Background { skyColor [ 0.2 0.2 0.2 ] }
    DirectionalLight { direction -1 -1 -1 }
    Solid {
      translation 0 0 -0.05
      children [ Shape { appearance PBRAppearance { baseColor 0.5 0.5 0.5 } geometry Box { size 20 20 0.1 } } ]
      boundingObject Box { size 20 20 0.1 }
    }
    '''
        world_info = re.search(r'^WorldInfo\s*\{.*?^\}', world.read_text(), re.M | re.S).group()
        header = header.replace('WorldInfo { basicTimeStep 10 }', world_info)
        (worlds/'test.wbt').write_text(header+model_text+'\n')
        cli = ['--world', str(world), '--duration', str(args.duration), '--scenario', args.scenario, '--velocity', str(args.velocity), '--log', str(run/'telemetry.jsonl')]
        if args.config:
            cli += ['--config', str(args.config.resolve())]
        wrapper = f'''import sys, traceback, json
    sys.path.insert(0, {str(root)!r})
    import wb_mpc_webots as app
    from controller import Supervisor
    instances = []
    class Probe(Supervisor):
        def __init__(self):
            super().__init__()
            instances.append(self)
            def find_tool(node):
                name = node.getField('name')
                if hasattr(name, 'type') and name.getSFString() == 'gripper_base':
                    return node
                children = node.getField('children')
                nodes = [children.getMFNode(i) for i in range(children.getCount())] if hasattr(children, 'type') else []
                endpoint = node.getField('endPoint')
                if hasattr(endpoint, 'type') and endpoint.getSFNode():
                    nodes.append(endpoint.getSFNode())
                for child in nodes:
                    found = find_tool(child)
                    if found:
                        return found
            self.tool = find_tool(self.getSelf())
            assert self.tool is not None
            self.truth = open({str(run/'truth.jsonl')!r}, 'w', buffering=1)
            self.next_truth = 0.
        def step(self, duration):
            status = super().step(duration)
            if status != -1 and self.getTime() >= self.next_truth:
                pos = app.np.array(self.tool.getPosition())
                rotation = app.np.array(self.tool.getOrientation()).reshape(3, 3)
                tip = pos + rotation @ app.np.array([0., 0., 0.1358])
                self.truth.write(json.dumps(dict(time=self.getTime(), tip=tip.tolist()))+'\\n')
                self.next_truth = self.getTime()+0.02
            return status
    app.load_robot_class = lambda: Probe
    code = 0
    try:
        app.main({cli!r})
    except Exception:
        with open({str(run/'error.txt')!r}, 'w') as error_file:
            error_file.write(traceback.format_exc())
        traceback.print_exc()
        code = 1
    finally:
        if instances:
            instances[0].simulationQuit(code)
    '''
        (controllers/'probe.py').write_text(inspect.cleandoc(wrapper)+'\n')
        (controllers/'runtime.ini').write_text('[python]\nCOMMAND = '+sys.executable+'\n')
        webots = Path(os.environ.get('WEBOTS_HOME', '/usr/local/webots'))/'webots'
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['WEBOTS_DISABLE_SAVE_SCREEN_PERSPECTIVE_ON_CLOSE'] = 'TRUE'
        command = [str(webots), '--batch', '--mode=fast', '--minimize', '--no-rendering', '--stdout', '--stderr', str(worlds/'test.wbt')]
        from wb_mpc_webots import arguments
        _, settings = arguments(cli)
        manifest = {'scenario': args.scenario, 'duration': args.duration, 'velocity': args.velocity,
                    'settings': asdict(settings), 'basic_time_step_ms': 10, 'python': sys.executable,
                    'world_sha256': hashlib.sha256(world.read_bytes()).hexdigest(),
                    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(root.glob('*.py'))}}
        (run/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('Test files:', run, flush=True)
        with (run/'webots.log').open('w') as out:
            process = subprocess.Popen(command, env=env, stdout=out, stderr=subprocess.STDOUT)
            try:
                status = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                raise RuntimeError(f'Webots timed out; inspect {run / "webots.log"}')
        print((run/'webots.log').read_text()[-5000:])
        if status or not (run/'telemetry.jsonl').exists():
            raise RuntimeError(f'Webots failed (exit {status}); inspect {run}')
    rows = [json.loads(line) for line in (run/'telemetry.jsonl').read_text().splitlines()]
    states = [r for r in rows if r['event'] == 'state']
    solves = [r for r in rows if r['event'] == 'solve']
    assert states and solves and rows[-1]['event'] == 'summary', 'Incomplete physical test'
    import numpy as np
    heights = [r['q'][2] for r in states]
    tilt = max(max(abs(x) for x in r['rpy'][:2]) for r in states)
    failure_fraction = sum(not r['ok'] for r in solves)/len(solves)
    # A hold-only fallback is not accepted as a successful MPC test.
    torque_fraction = sum(r['mode'] == 'torque' for r in states)/len(states)
    from wb_mpc_webots import arguments
    _, settings = arguments(['--config', str(args.config)] if args.config else [])
    if (run/'manifest.json').exists():
        from mpc_bridge import Settings
        settings = Settings.from_metadata(json.loads((run/'manifest.json').read_text())['settings'])
    result = {'scenario': args.scenario, 'duration': args.duration, 'min_height': min(heights),
              'max_height': max(heights), 'max_tilt': tilt, 'failure_fraction': failure_fraction,
              'torque_fraction': torque_fraction, 'solves': len(solves),
              'elapsed_p95': float(np.percentile([r['elapsed'] for r in solves], 95)),
              'deadline_misses': sum(r['elapsed'] > settings.solve_period for r in solves),
              'passed': min(heights) > 0.20 and max(heights) < 0.5 and tilt < 0.5 and failure_fraction < 0.05 and torque_fraction > 0.95}
    result['faults'] = sum(r['event'] == 'fault' for r in rows)
    result['passed'] &= result['faults'] == 0
    if len(solves) > 1:
        intervals = np.diff([r['time'] for r in solves])
        result['solve_interval_min'] = float(intervals.min())
        result['solve_interval_max'] = float(intervals.max())
        # Gait switches intentionally request an immediate solve. The manual
        # and arm scenarios never switch gait, so their cadence is testable.
        if args.scenario != 'trot':
            manifest_path = run/'manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            feedback_dt = manifest.get('basic_time_step_ms', 2)/1000.
            aligned = abs(settings.solve_period/feedback_dt-round(settings.solve_period/feedback_dt)) < 1e-8
            tolerance = 1e-8 if aligned else feedback_dt+1e-8
            result['cadence_passed'] = bool(np.all(np.abs(intervals-settings.solve_period) <= tolerance))
            result['passed'] &= result['cadence_passed']
    if args.scenario == 'trot':
        moving = [r for r in states if r['gait'] == 'trot']
        patterns = {tuple(int(v > 0) for v in r['contacts']) for r in moving}
        result['both_diagonal_contacts'] = {(1, 0, 0, 1), (0, 1, 1, 0)} <= patterns
        result['leg_motion_range'] = float(np.ptp(np.array([r['q'][7:19] for r in moving]), axis=0).max()) if moving else 0.
        result['passed'] &= result['both_diagonal_contacts'] and result['leg_motion_range'] > 0.05
        if args.velocity:
            result['forward_displacement'] = moving[-1]['q'][0]-moving[0]['q'][0]
            result['passed'] &= result['forward_displacement']*np.sign(args.velocity) > 0.01
    if args.scenario == 'arm':
        from world_model import WebotsModel
        import pinocchio as pin
        model = WebotsModel()
        def tool_z(row):
            pin.framesForwardKinematics(model.model, model.data, np.array(row['q']))
            return model.data.oMf[model.arm_ee_frame].translation[2]
        before = min(states, key=lambda r: abs(r['time']-3.))
        after = min(states, key=lambda r: abs(r['time']-4.))
        result['tool_height_change'] = float(tool_z(after)-tool_z(before))
        result['passed'] &= 0.003 < result['tool_height_change'] < 0.03
        if (run/'truth.jsonl').exists():
            truth = [json.loads(line) for line in (run/'truth.jsonl').read_text().splitlines()]
            actual_before = min(truth, key=lambda r: abs(r['time']-before['time']))
            actual_after = min(truth, key=lambda r: abs(r['time']-after['time']))
            result['actual_tool_height_change'] = actual_after['tip'][2]-actual_before['tip'][2]
            result['tool_fk_height_error'] = max(abs(tool_z(row)-min(truth, key=lambda r: abs(r['time']-row['time']))['tip'][2]) for row in states)
            result['passed'] &= 0.003 < result['actual_tool_height_change'] < 0.03
            result['passed'] &= result['tool_fk_height_error'] < 0.003
    result['passed'] = bool(result['passed'])
    (run/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise RuntimeError('Physical test acceptance failed')


if __name__ == '__main__':
    main()
