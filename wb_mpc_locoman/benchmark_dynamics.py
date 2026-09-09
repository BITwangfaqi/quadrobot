"""Headless solve-time comparison using the settings in main.py."""

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import platform
import time
from datetime import datetime, timezone

import casadi as ca
import numpy as np
import pinocchio as pin

import main as config
from args import DYN_ARGS, SOLVER_ARGS
from optimization import make_ocp
from utils.robot import B2_Z1


# Table I has six configurations, rather than one entry per Dynamics class.
# Keep the original model names for the include_base=True cases and CLI compatibility.
CASES = {
    "whole_body_rnea": ("whole_body_rnea", {"include_acc": True}),
    "whole_body_aba": ("whole_body_aba", {}),
    "centroidal_vel": ("centroidal_vel", {"include_base": True}),
    "centroidal_vel_no_base": ("centroidal_vel", {"include_base": False}),
    "centroidal_acc": ("centroidal_acc", {"include_base": True}),
    "centroidal_acc_no_base": ("centroidal_acc", {"include_base": False}),
    "whole_body_acc": ("whole_body_acc", {"include_base": True}),
}
MODELS = tuple(CASES)[:6]


def case_settings(case):
    dynamics, overrides = CASES[case]
    dyn_args = copy.deepcopy(DYN_ARGS[dynamics])
    dyn_args.update(overrides)
    return dynamics, dyn_args


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def run_model(model, loops, repeat, discard, solver):
    start = time.perf_counter()
    dynamics, dyn_args = case_settings(model)
    robot = B2_Z1(reference_pose="standing_with_arm_up", arm_joints=4)
    robot.set_gait_sequence(config.gait_type, config.gait_period)
    pin.computeAllTerms(robot.model, robot.data, robot.q0, np.zeros(robot.model.nv))
    ocp = make_ocp(
        dynamics=dynamics, dyn_args=dyn_args, robot=robot,
        nodes=config.nodes, tau_nodes=config.tau_nodes, warm_start=config.warm_start,
    )
    ocp.set_time_params(config.dt_min, config.dt_max)
    ocp.set_swing_params(config.swing_height, config.swing_vel_limits)
    ocp.set_tracking_targets(config.base_vel_des, config.arm_vel_des, config.arm_force_des)
    x_init = ocp.x_nom
    ocp.update_params(x_init, 0)
    ocp.init_solver(solver, copy.deepcopy(SOLVER_ARGS[solver]))
    setup_s = time.perf_counter() - start
    print(f"{model}, repeat {repeat}: setup {setup_s:.2f} s", flush=True)
    rows = []
    for k in range(loops):
        ocp.update_params(x_init, k * config.dt_min)
        params = ocp.get_solver_params()
        start = time.perf_counter_ns()
        sol_x = ocp.solver_function(*params)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        if not np.isfinite(np.asarray(sol_x)).all():
            raise RuntimeError(f"{model}: non-finite solution at iteration {k}")
        g, lbg, ubg = ocp.g_data(sol_x, ocp.opti.value(ocp.opti.p))
        cv = float(ocp.constr_viol_norm_inf(g, lbg, ubg))
        if not np.isfinite(cv):
            raise RuntimeError(f"{model}: non-finite constraint violation at iteration {k}")
        rows.append({
            "model": model, "repeat": repeat, "iteration": k,
            "time_s": k * config.dt_min, "solve_ms": elapsed_ms,
            "constraint_violation_inf": cv, "retained": k >= discard,
        })
        ocp.retract_stacked_sol(sol_x, retract_all=False)
        x_init = ocp.dyn.state_integrate()(x_init, ocp.DX_prev[1])
        if (k + 1) % 50 == 0 or k + 1 == loops:
            print(f"  {k + 1}/{loops}: {elapsed_ms:.3f} ms, CV {cv:.3g}", flush=True)
    details = {
        "model": model, "repeat": repeat, "setup_s": setup_s,
        "dynamics": dynamics, "dynamics_args": dyn_args,
        "decision_variables": int(ocp.opti.nx), "constraints": int(ocp.opti.ng),
        "horizon_s": float(sum(ocp.opti.value(dt) for dt in ocp.dts)),
        "first_solve_ms": rows[0]["solve_ms"],
    }
    return rows, details


def write_results(output, rows, runs, metadata):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for model in metadata["models"]:
        selected = [row for row in rows if row["model"] == model]
        if not selected:
            continue
        retained = [row for row in selected if row["retained"]]
        tolerance = SOLVER_ARGS[metadata["solver"]]["opts"][metadata["solver"] + ".tol"]
        summaries.append({
            "model": model, "samples": len(selected), "retained_samples": len(retained),
            "all_solve_ms": distribution([r["solve_ms"] for r in selected]),
            "retained_solve_ms": distribution([r["solve_ms"] for r in retained]),
            "constraint_violation_inf": distribution([r["constraint_violation_inf"] for r in selected]),
            "cv_within_tolerance_count": sum(r["constraint_violation_inf"] <= tolerance for r in selected),
        })
    (output / "results.json").write_text(
        json.dumps({"metadata": metadata, "runs": runs, "summary": summaries}, indent=2) + "\n"
    )
    lines = [
        "# Dynamics solve-time benchmark", "",
        f"Recorded (UTC): {metadata['created_utc']}. CPU: {metadata['cpu']}.", "",
        f"Solver: {metadata['solver']}; nodes: {config.nodes}; tau_nodes: {config.tau_nodes}; "
        f"loops per repeat: {metadata['loops']}; repeats: {metadata['repeats']}; "
        f"warm start: {config.warm_start}; compiled: False.", "",
        "Times measure only the solver_function call with perf_counter_ns. "
        "Setup, parameter updates, constraint checks, solution retraction and visualization are excluded. "
        "Each run starts from the nominal pose and propagates its own predicted next state as in main.py. "
        "Model-specific weights and constraints remain at their repository defaults. "
        "This matches the six formulations of Table I, but does not reproduce its "
        "separate Pinocchio forward-dynamics simulation or compiled benchmarks.", "",
        "| Configuration | OCP dynamics | Dynamics arguments | Decision variables |",
        "| --- | --- | --- | ---: |",
    ]
    for model in metadata["models"]:
        details = next((run for run in runs if run["model"] == model), None)
        if details is not None:
            lines.append(f"| {model} | {details['dynamics']} | "
                         f"`{json.dumps(details['dynamics_args'])}` | {details['decision_variables']} |")
    lines += ["",
        "| Model | N | Mean (ms) | Std (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        stats = summary["all_solve_ms"]
        lines.append(f"| {summary['model']} | {summary['samples']} | " + " | ".join(
            f"{stats[key]:.3f}" for key in ("mean", "std", "median", "p95", "min", "max")
        ) + " |")
    lines += ["", f"After excluding the first {metadata['discard']} iterations of each run:", "",
              "| Model | N | Mean (ms) | Std (ms) | P95 (ms) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for summary in summaries:
        stats = summary["retained_solve_ms"]
        lines.append(f"| {summary['model']} | {summary['retained_samples']} | "
                     f"{stats['mean']:.3f} | {stats['std']:.3f} | {stats['p95']:.3f} |")
    lines += ["", "| Model | Mean CV (inf) | Max CV (inf) | CV ≤ solver tolerance |",
              "| --- | ---: | ---: | ---: |"]
    for summary in summaries:
        cv = summary["constraint_violation_inf"]
        lines.append(f"| {summary['model']} | {cv['mean']:.6g} | {cv['max']:.6g} | "
                     f"{summary['cv_within_tolerance_count']}/{summary['samples']} |")
    lines += ["", "Constraint feasibility alone does not establish solver convergence or optimality. "
              f"The {metadata['solver']} iteration budget is "
              f"{metadata['solver_args']['opts'][metadata['solver'] + '.max_iter']}. "
              "RNEA represents the torque Dynamics class; ABA is an alternative OCP using that same class.", ""]
    report = "\n".join(lines)
    (output / "summary.md").write_text(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", "--models", dest="models", nargs="+", choices=tuple(CASES),
                        default=list(MODELS), help="Table I configurations; --models is a compatibility alias")
    parser.add_argument("--loops", type=int, default=config.mpc_loops)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--discard", type=int, default=10, help="Exclude initial iterations in additional statistics only")
    parser.add_argument("--solver", choices=("fatrop", "ipopt"), default=config.solver)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "benchmark_results_six_configs")
    args = parser.parse_args()
    if args.loops < 1 or args.repeats < 1 or not 0 <= args.discard < args.loops:
        parser.error("Require loops > 0, repeats > 0, and 0 <= discard < loops")
    if len(set(args.models)) != len(args.models):
        parser.error("Configurations must not be repeated; use --repeats instead")
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu, "platform": platform.platform(), "python": platform.python_version(),
        "casadi": ca.__version__, "pinocchio": pin.__version__, "numpy": np.__version__,
        "models": args.models, "solver": args.solver, "solver_args": SOLVER_ARGS[args.solver],
        "dynamics_models": {model: case_settings(model)[0] for model in args.models},
        "dynamics_args": {model: case_settings(model)[1] for model in args.models},
        "loops": args.loops, "repeats": args.repeats, "discard": args.discard,
        "robot": "B2_Z1", "arm_joints": 4, "reference_pose": "standing_with_arm_up",
        "nodes": config.nodes, "tau_nodes": config.tau_nodes,
        "state_nodes": config.nodes + 1,
        "dt_min": config.dt_min, "dt_max": config.dt_max,
        "gait_type": config.gait_type, "gait_period": config.gait_period,
        "swing_height": config.swing_height, "swing_vel_limits": config.swing_vel_limits,
        "base_vel_des": config.base_vel_des.tolist(), "arm_vel_des": config.arm_vel_des.tolist(),
        "arm_force_des": config.arm_force_des.tolist(), "warm_start": config.warm_start,
        "compiled": False, "timing_scope": "solver_function wall time only",
        "state_update": "predicted next state, as in main.py; no separate forward-dynamics simulation",
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    rows, runs = [], []
    # Sequential execution avoids competing benchmark solvers on the same CPU.
    for repeat in range(1, args.repeats + 1):
        for model in args.models:
            samples, details = run_model(model, args.loops, repeat, args.discard, args.solver)
            rows.extend(samples)
            runs.append(details)
            report = write_results(args.output, rows, runs, metadata)
            gc.collect()
    print(report)
    print(f"Results saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
