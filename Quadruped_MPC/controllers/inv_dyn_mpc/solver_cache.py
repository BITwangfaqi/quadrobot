"""Compile the complete Fatrop entry point and trajectory extraction, as in the paper."""
from dataclasses import asdict
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import numpy as np


class CompiledSolver:
    def __init__(self, ca, path, symbolic):
        self.ca, self.path, self.symbolic = ca, str(path), symbolic
        # Load dependencies with ordinary dlopen before CasADi's deep binding.
        # Otherwise libomp initialization can resolve libc globals incorrectly.
        self.library = ctypes.CDLL(str(path))
        self.function = ca.external(symbolic.name(), str(path))
        self.library.inv_dyn_stats.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.library.inv_dyn_stats.restype = None
        self._stats = {}

    def __call__(self, **kwargs):
        result = self.function(**kwargs)
        values = (ctypes.c_double * 11)()
        self.library.inv_dyn_stats(values)
        profile = dict(zip(('compute_sd_time', 'duinf_time', 'eval_hess_time', 'eval_jac_time',
                            'eval_cv_time', 'eval_grad_time', 'eval_obj_time',
                            'initialization_time', 'time_total'), values[:9]))
        self._stats = dict(success=int(values[10]) == 0, return_status=str(int(values[10])),
                           iter_count=int(values[9]), fatrop=profile)
        return result

    def stats(self):
        return self._stats

    def get_function(self, name):
        return self.ca.external(name, self.path)


def compiled_solver(ca, nlp, options, model, config, retraction):
    if config.solver != 'fatrop':
        return ca.nlpsol('whole_body_inverse_dynamics', config.solver, nlp, options), retraction, False
    root = Path(__file__).resolve().parent
    compiler = os.environ.get('CC') or shutil.which('clang')
    if not compiler and Path('/usr/lib/llvm-14/bin/clang').exists():
        compiler = '/usr/lib/llvm-14/bin/clang'
    compiler = compiler or shutil.which('gcc')
    if not compiler:
        raise RuntimeError('C code generation needs clang or gcc; use jit=false for diagnostics')
    prefix = Path(sys.prefix)
    include, library = prefix / 'include', prefix / 'lib'
    openmp = None
    if config.derivative_threads > 1:
        for candidate in (Path('/usr/lib/llvm-14'), root / '.cache/toolchain/usr/lib/llvm-14'):
            if (candidate / 'lib/libomp.so').exists():
                openmp = candidate
                break
        if openmp is None and 'clang' in compiler:
            raise RuntimeError('Parallel C derivatives need libomp; install libomp-14-dev or set derivative_threads=1')
    if not (include / 'fatrop/ocp/OCPCInterface.h').exists():
        raise RuntimeError('Complete Fatrop compilation needs Fatrop/BLASFEO headers and libraries; '
                           'use the configured wewebot environment or jit=false')
    symbolic = ca.nlpsol('whole_body_inverse_dynamics', config.solver, nlp, options)
    def encode(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, Path): return str(value)
        raise TypeError(type(value).__name__)
    digest = hashlib.sha256()
    for name in ('mpc.py', 'rigid_body_model.py', 'reduced_model.py', 'robot_description.py', 'solver_cache.py'):
        digest.update((root / name).read_bytes())
    metadata = dict(casadi=ca.__version__, prefix=str(prefix), compiler=compiler,
                    machine=platform.machine(), cpu=platform.processor(), config=asdict(config),
                    model=asdict(model.description), backend=model.casadi_backend)
    digest.update(json.dumps(metadata, sort_keys=True, default=encode).encode())
    directory = root / '.cache' / digest.hexdigest()[:24]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'solver.so'
    with (directory / 'build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        hit = target.exists()
        if not hit:
            started = time.perf_counter()
            print(f'[inv_dyn_mpc] Compiling complete Fatrop solver: {directory}', flush=True)
            generator = ca.CodeGenerator('solver.c')
            generator.add(symbolic)
            generator.add(retraction)
            for name in ('nlp_f', 'nlp_g', 'nlp_grad_f', 'nlp_jac_g', 'nlp_hess_l'):
                generator.add(symbolic.get_function(name))
            generator.generate(str(directory) + '/')
            source = directory / 'solver.c'
            code = source.read_text()
            # Retain actual convergence/iteration diagnostics from generated C.
            marker = 'static struct casadi_fatrop_data casadi_f0_mem[CASADI_MAX_NUM_THREADS];'
            if code.count(marker) != 1:
                raise RuntimeError('Unsupported CasADi generated Fatrop memory layout')
            # CasADi 3.7 generates a post-solve nlp_grad call even when both
            # multiplier calculations are disabled. That reverse sweep still
            # evaluates unused parameter derivatives. Recompute only f and g,
            # preserving the exact returned objective and feasibility values.
            postsolve = re.compile(
                r'(  d->res\[2\] = 0;\n  d->res\[3\] = 0;\n)'
                r'  if \(casadi_f\d+\(d->arg, d->res, d->iw, d->w, 0\)\) return 1;')
            replacement = r'''\1  if (p.nlp_f.eval(d->arg, d->res, d->iw, d->w, 0)) return 1;
  d->res[0] = d_nlp.z + p_nlp.nx;
  if (p.nlp_g.eval(d->arg, d->res, d->iw, d->w, 0)) return 1;'''
            code, count = postsolve.subn(replacement, code)
            if count != 1 or options.get('calc_lam_p', True) or options.get('calc_lam_x', True):
                raise RuntimeError('Unsupported generated post-solve callback layout')
            if config.derivative_threads > 1:
                # Fix the team size in generated loops; do not depend on the
                # BLAS/OMP environment used by the Python controller.
                code = code.replace('#pragma omp parallel for ',
                                    f'#pragma omp parallel for num_threads({config.derivative_threads}) ')
            footer = '''
CASADI_SYMBOL_EXPORT void inv_dyn_stats(double* out) {
  const struct casadi_fatrop_data* d = &casadi_f0_mem[0];
  out[0]=d->stats.compute_sd_time; out[1]=d->stats.duinf_time;
  out[2]=d->stats.eval_hess_time; out[3]=d->stats.eval_jac_time;
  out[4]=d->stats.eval_cv_time; out[5]=d->stats.eval_grad_time;
  out[6]=d->stats.eval_obj_time; out[7]=d->stats.initialization_time;
  out[8]=d->stats.time_total; out[9]=d->stats.iterations_count;
  out[10]=d->return_status;
}
'''
            source.write_text(code + footer)
            pending = directory / 'solver.pending.so'
            command = [compiler, '-O3', '-march=native', '-fPIC', '-shared', str(source),
                       '-I' + str(include), '-L' + str(library), '-Wl,-rpath,' + str(library),
                       '-lfatrop', '-lblasfeo', '-lm', '-o', str(pending)]
            if config.derivative_threads > 1:
                command += ['-fopenmp']
                if openmp is not None:
                    command += ['-I' + str(openmp / 'lib/clang/14.0.0/include'),
                                '-L' + str(openmp / 'lib'), '-Wl,-rpath,' + str(openmp / 'lib')]
            with (directory / 'compiler.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            pending.replace(target)
            (directory / 'build.json').write_text(json.dumps(metadata, default=encode, indent=2))
            print(f'[inv_dyn_mpc] Complete solver cached in {time.perf_counter() - started:.1f}s', flush=True)
    return CompiledSolver(ca, target, symbolic), ca.external('retract_solution', str(target)), hit
