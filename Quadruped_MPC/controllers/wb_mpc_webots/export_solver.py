"""Generate the original Fatrop C solver and its Webots trajectory decoder."""
import argparse
from dataclasses import asdict
import json
import os
import hashlib
import time
from pathlib import Path
import subprocess
import sys

import casadi as ca
import numpy as np

from world_model import WebotsModel
from mpc_bridge import MPC, Settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'.cache/export')
    parser.add_argument('--nominal', type=Path, help='JSON measured settled generalized position')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--reuse-generated', action='store_true', help='compile existing C files after checking their metadata')
    parser.add_argument('--optimization', choices=('0', '1', '2', '3'), default='0',
                        help='GCC optimization level; large generated functions make higher levels expensive to compile')
    args = parser.parse_args()
    settings = Settings(**(json.loads(args.config.read_text()) if args.config else {}))
    if settings.solver != 'fatrop' or settings.compiled_solver:
        parser.error('Export requires solver=fatrop and an empty compiled_solver')
    robot = WebotsModel()
    if args.nominal:
        robot.set_nominal(np.asarray(json.loads(args.nominal.read_text()), float))
    mpc = MPC(robot, settings)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    metadata = {'settings': asdict(settings), 'model_signature': robot.signature(),
                'nominal': robot.q0.tolist(), 'casadi': ca.__version__,
                'input_names': mpc.ocp.solver_function.name_in(),
                'functions': ['solver_function', 'webots_decode', 'g_data'],
                'variables': mpc.ocp.opti.nx, 'constraints': mpc.ocp.opti.ng}
    if args.reuse_generated:
        existing = json.loads((output/'solver_function.json').read_text())
        existing['settings'] = json.loads(json.dumps(asdict(Settings.from_metadata(existing['settings']))))
        for key, value in json.loads(json.dumps(metadata)).items():
            if existing.get(key) != value:
                raise ValueError(f'Existing generated code metadata mismatch: {key}')
    else:
        try:
            os.chdir(output)
            # Full solver export is the original Opti.to_function codegen path.
            mpc.ocp.solver_function.generate('solver_function.c')
            mpc.decoder.generate('webots_decode.c')
            mpc.ocp.g_data.generate('g_data.c')
        finally:
            os.chdir(previous)
    (output/'solver_function.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print('Generated solver and decoder in', output, flush=True)
    if args.compile:
        prefix = Path(sys.prefix)
        cmd = [os.environ.get('CC', 'cc'), f'-O{args.optimization}', '-fPIC', '-shared',
               '-I'+str(prefix/'include'), str(output/'solver_function.c'),
               str(output/'webots_decode.c'), str(output/'g_data.c'), '-L'+str(prefix/'lib'),
               '-Wl,-rpath,'+str(prefix/'lib'), '-lfatrop', '-lblasfeo', '-lm',
               '-o', str(output/'solver_function.so')]
        print('Compiling full MPC (this can take several minutes)...', flush=True)
        started = time.perf_counter()
        subprocess.run(cmd, check=True)
        metadata['compile_seconds'] = time.perf_counter()-started
        metadata['compiler_command'] = cmd
        compiled = ca.external('solver_function', str(output/'solver_function.so'))
        values = mpc.ocp.get_solver_params()
        expected = mpc.ocp.solver_function(*values)
        actual = compiled(*values)
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
        parameters = mpc.ocp.opti.value(mpc.ocp.opti.p)
        for name, function in [('webots_decode', mpc.decoder), ('g_data', mpc.ocp.g_data)]:
            external = ca.external(name, str(output/'solver_function.so'))
            for native, built in zip(function(expected, parameters), external(expected, parameters)):
                np.testing.assert_allclose(built, native, atol=1e-7, rtol=1e-7)
        metadata['library_sha256'] = hashlib.sha256((output/'solver_function.so').read_bytes()).hexdigest()
        metadata['equivalence_passed'] = True
        (output/'solver_function.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('PASS compiled solver matches the original interpreted function', flush=True)


if __name__ == '__main__':
    main()
