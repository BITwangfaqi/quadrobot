# Dynamics solve-time benchmark

Recorded (UTC): 2026-09-09T14:55:54.297712+00:00. CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz.

Solver: fatrop; nodes: 14; tau_nodes: 3; loops per repeat: 200; repeats: 1; warm start: True; compiled: False.

Times measure only the solver_function call with perf_counter_ns. Setup, parameter updates, constraint checks, solution retraction and visualization are excluded. Each run starts from the nominal pose and propagates its own predicted next state as in main.py. Model-specific weights and constraints remain at their repository defaults. This matches the six formulations of Table I, but does not reproduce its separate Pinocchio forward-dynamics simulation or compiled benchmarks.

| Configuration | OCP dynamics | Dynamics arguments | Decision variables |
| --- | --- | --- | ---: |
| whole_body_rnea | whole_body_rnea | `{"include_acc": true}` | 1226 |
| whole_body_aba | whole_body_aba | `{}` | 1094 |
| centroidal_vel | centroidal_vel | `{"include_base": true}` | 938 |
| centroidal_vel_no_base | centroidal_vel | `{"include_base": false}` | 854 |
| centroidal_acc | centroidal_acc | `{"include_base": true}` | 1178 |
| centroidal_acc_no_base | centroidal_acc | `{"include_base": false}` | 1094 |

| Model | N | Mean (ms) | Std (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| whole_body_rnea | 200 | 46.047 | 10.735 | 39.861 | 69.821 | 38.375 | 91.088 |
| whole_body_aba | 200 | 138.100 | 53.214 | 111.849 | 262.909 | 104.374 | 364.084 |
| centroidal_vel | 200 | 64.699 | 30.511 | 58.282 | 148.153 | 45.613 | 167.932 |
| centroidal_vel_no_base | 200 | 222.315 | 91.595 | 192.946 | 465.051 | 146.791 | 516.000 |
| centroidal_acc | 200 | 189.197 | 50.449 | 161.419 | 298.755 | 151.457 | 396.302 |
| centroidal_acc_no_base | 200 | 388.314 | 162.282 | 359.347 | 870.677 | 269.210 | 927.910 |

After excluding the first 10 iterations of each run:

| Model | N | Mean (ms) | Std (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| whole_body_rnea | 190 | 46.143 | 10.897 | 69.891 |
| whole_body_aba | 190 | 138.945 | 54.281 | 278.677 |
| centroidal_vel | 190 | 65.438 | 31.103 | 148.365 |
| centroidal_vel_no_base | 190 | 224.475 | 93.373 | 466.421 |
| centroidal_acc | 190 | 189.817 | 51.153 | 299.371 |
| centroidal_acc_no_base | 190 | 390.654 | 165.709 | 870.793 |

| Model | Mean CV (inf) | Max CV (inf) | CV ≤ solver tolerance |
| --- | ---: | ---: | ---: |
| whole_body_rnea | 1.96059e-07 | 3.3609e-06 | 200/200 |
| whole_body_aba | 3.15148 | 272.764 | 197/200 |
| centroidal_vel | 0.0161585 | 1.19533 | 195/200 |
| centroidal_vel_no_base | 3.28238e-05 | 0.00558289 | 199/200 |
| centroidal_acc | 2.10362e-07 | 3.3565e-06 | 200/200 |
| centroidal_acc_no_base | 0.0180556 | 1.45471 | 197/200 |

Constraint feasibility alone does not establish solver convergence or optimality. The fatrop iteration budget is 10. RNEA represents the torque Dynamics class; ABA is an alternative OCP using that same class.
