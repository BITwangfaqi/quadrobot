# Dynamics solve-time benchmark

Recorded (UTC): 2026-09-09T14:35:15.214720+00:00. CPU: Intel(R) Core(TM) i9-9900 CPU @ 3.10GHz.

Solver: fatrop; nodes: 14; tau_nodes: 3; loops per repeat: 200; repeats: 1; warm start: True; compiled: False.

Times measure only the solver_function call with perf_counter_ns. Setup, parameter updates, constraint checks, solution retraction and visualization are excluded. Each run starts from the nominal pose and propagates its own predicted next state as in main.py. Model-specific weights and constraints remain at their repository defaults.

| Model | N | Mean (ms) | Std (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| centroidal_vel | 200 | 64.058 | 29.614 | 57.732 | 144.356 | 45.253 | 166.169 |
| centroidal_acc | 200 | 178.888 | 48.750 | 150.074 | 287.487 | 146.472 | 381.687 |
| whole_body_acc | 200 | 48.020 | 11.071 | 41.671 | 72.272 | 39.895 | 94.425 |
| whole_body_rnea | 200 | 45.957 | 10.709 | 39.818 | 70.003 | 38.516 | 91.200 |

After excluding the first 10 iterations of each run:

| Model | N | Mean (ms) | Std (ms) | P95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| centroidal_vel | 190 | 64.714 | 30.213 | 144.415 |
| centroidal_acc | 190 | 179.408 | 49.478 | 287.586 |
| whole_body_acc | 190 | 48.120 | 11.234 | 72.336 |
| whole_body_rnea | 190 | 46.088 | 10.864 | 70.104 |

| Model | Mean CV (inf) | Max CV (inf) | CV ≤ solver tolerance |
| --- | ---: | ---: | ---: |
| centroidal_vel | 0.0161585 | 1.19533 | 195/200 |
| centroidal_acc | 2.10362e-07 | 3.3565e-06 | 200/200 |
| whole_body_acc | 2.04223e-07 | 3.3878e-06 | 200/200 |
| whole_body_rnea | 1.96059e-07 | 3.3609e-06 | 200/200 |

Constraint feasibility alone does not establish solver convergence or optimality. The default Fatrop iteration budget is 10, so these timings are for that budget. RNEA represents the torque Dynamics class; ABA is an alternative OCP using that same class.
