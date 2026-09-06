"""Run the actual controller in a private, headless copy of the Webots world.

Requires Webots and Xvfb. The user's world and running simulator are untouched.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep", type=int, default=2)
    parser.add_argument("--duration", type=float, default=6.)
    parser.add_argument("--xvfb", default="/tmp/inv_dyn_xvfb/usr/bin/Xvfb")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--controller-root", type=Path)
    parser.add_argument("--scenario", choices=('stand', 'trot', 'tool'), default='stand')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    webots = Path(os.environ.get("WEBOTS_HOME", "/usr/local/webots"))
    original = root.parents[1] / "worlds" / "quadruped_arm.wbt"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    project = output / "project"
    worlds = project / "worlds"
    controller = project / "controllers" / "inv_dyn_probe"
    worlds.mkdir(parents=True, exist_ok=True)
    controller.mkdir(parents=True, exist_ok=True)
    if not (project / 'protos').exists():
        (project / 'protos').symlink_to(root.parents[1] / 'protos', target_is_directory=True)
    source = original.read_text()
    start = re.search(r"(?m)^Robot\s*\{", source).start()
    tokens = re.finditer(r'"(?:\\.|[^"\\])*"|#[^\n]*|[{}]', source[start:])
    depth = 0
    for token in tokens:
        if token.group() == "{": depth += 1
        elif token.group() == "}":
            depth -= 1
            if depth == 0:
                robot_source = source[start:start + token.end()]
                break
    robot_source = robot_source.replace('controller "<extern>"', 'controller "inv_dyn_probe"\n  supervisor TRUE')
    # Explicit flat floor avoids external PROTO downloads affecting this test.
    world = (f'#VRML_SIM R2023a utf8\nWorldInfo {{ basicTimeStep {args.timestep} }}\n'
             'Viewpoint { position 2 2 1 }\n'
             'Solid { translation 0 0 -0.05 children [ Shape { geometry Box { size 80 1.5 0.1 } } ] '
             'boundingObject Box { size 80 1.5 0.1 } }\n' + robot_source)
    (worlds / "probe.wbt").write_text(world)
    command = ["--world", str(original), "--duration", str(args.duration)]
    if args.config:
        command += ["--config", str(args.config.resolve())]
    script = f'''import sys, json, math
sys.path.insert(0, {str(args.controller_root.resolve() if args.controller_root else root)!r})
sys.stdout = sys.stderr = open({str(output / 'controller.log')!r}, 'w', buffering=1)
import inv_dyn_mpc as app
from controller import Supervisor
log = open({str(output / 'states.jsonl')!r}, 'w', buffering=1)
instance = None
class Probe(Supervisor):
    failed = False
    def __init__(self):
        global instance
        super().__init__()
        instance = self
    def step(self, timestep):
        result = super().step(timestep)
        if result != -1:
            node = self.getSelf()
            p = node.getPosition()
            velocity = node.getVelocity()
            joints = [self.getDevice(name).getPositionSensor().getValue() for name in
                      tuple(f'{{leg}}_{{joint}}_motor' for leg in ('FR','FL','BR','BL') for joint in ('hip','leg','foot'))
                      + tuple(f'joint{{i}}' for i in range(1,7))]
            log.write(json.dumps(dict(t=self.getTime(), p=p, velocity=velocity, joints=joints)) + '\\n')
            if not all(math.isfinite(x) for x in p + velocity) or not .15 < p[2] < 1.0 or max(abs(x) for x in velocity) > 50:
                print('FAIL: unstable robot state', self.getTime(), p, velocity, flush=True)
                self.failed = True
                self.simulationQuit(2)
                return -1
        return result
app._load_webots_robot = lambda: Probe
# Record every complete solve call, not only the 1 Hz human-readable log.
import mpc, time
timings = open({str(output / 'solves.jsonl')!r}, 'w', buffering=1)
original_solve = mpc.InverseDynamicsMPC.solve
def timed_solve(self, *args):
    started = time.perf_counter()
    try:
        solution = original_solve(self, *args)
        timings.write(json.dumps(dict(t=instance.getTime(), wall_ms=1000*(time.perf_counter()-started),
                                     iterations=solution.iterations, violation=solution.constraint_violation,
                                     success=True)) + '\\n')
        return solution
    except Exception as error:
        timings.write(json.dumps(dict(t=instance.getTime(), wall_ms=1000*(time.perf_counter()-started),
                                     success=False, error=str(error))) + '\\n')
        raise
mpc.InverseDynamicsMPC.solve = timed_solve
class AutoCommand(app.UserCommand):
    def update(self, keyboard, now, dt, q):
        keys = []
        if now >= 5.:
            if {args.scenario!r} == 'trot': keys = [ord('U'), ord('W')]
            elif {args.scenario!r} == 'tool': keys = [ord('I')]
        iterator = iter(keys)
        class Keys:
            def getKey(self): return next(iterator, -1)
            def __getattr__(self, name): return getattr(keyboard, name)
        return super().update(Keys(), now, dt, q)
app.UserCommand = AutoCommand
failed = True
try:
    app.main({command!r})
    failed = instance.failed
finally:
    if instance:
        instance.simulationQuit(2 if failed else 0)
    log.close()
    timings.close()
'''
    (controller / "inv_dyn_probe.py").write_text(script)
    # Webots discovers the Python command through runtime.ini in the controller.
    (controller / "runtime.ini").write_text(f"[python]\nCOMMAND = {sys.executable}\n")
    env = dict(os.environ, WEBOTS_HOME=str(webots), OPENBLAS_NUM_THREADS="1",
               QTWEBENGINE_DISABLE_SANDBOX="1", LIBGL_ALWAYS_SOFTWARE="1")
    env.pop("QT_QPA_PLATFORM", None)
    read_fd, write_fd = os.pipe()
    xlog = (output / "xvfb.log").open("w")
    xserver = subprocess.Popen([args.xvfb, "-displayfd", str(write_fd), "-screen", "0", "1280x720x24",
                               "-nolisten", "tcp", "-ac"], pass_fds=(write_fd,), stdout=xlog, stderr=xlog)
    os.close(write_fd)
    try:
        with os.fdopen(read_fd) as pipe:
            display = pipe.readline().strip()
        if not display:
            raise RuntimeError(f"Xvfb failed; see {output / 'xvfb.log'}")
        env["DISPLAY"] = ":" + display
        with (output / "webots.log").open("w") as log:
            run = subprocess.run([str(webots / "webots"), "--batch", "--mode=fast", "--no-rendering", "--stdout", "--stderr",
                                  str(worlds / "probe.wbt")], env=env, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=900)
        print(f"Webots exit={run.returncode}; logs: {output}")
        if run.returncode:
            raise SystemExit(run.returncode)
    finally:
        xserver.terminate()
        xserver.wait(timeout=10)
        xlog.close()


if __name__ == "__main__":
    main()
