# 倒立摆正／逆动力学 MPC 耗时复测

记录时间（UTC）：2026-09-09T15:33:58.513775+00:00。CPU：Intel Core i9-9900。Python：/home/tdt/anaconda3/envs/wewebot/bin/python；CasADi：3.7.0；BLAS/OMP/MKL 各设为单线程。

两套控制器均使用 IPOPT，N=20 个控制区间、21 个状态节点，时间步 20–45 ms，max_iter=40、tol=1e-5、acceptable_tol=1e-4。保留原有轨迹热启动。

从正动力学已有离线日志每 10 行抽取一次状态，共 751 个状态，覆盖 15 s、间隔 20 ms。每种配置回放完全相同的状态 3 轮，交替正／逆动力学运行顺序，每轮使用独立进程，串行执行。此测试没有启动 Webots，也不是新的闭环仿真。

original 保留两个项目原配置；matched 仅在逆动力学测试进程中将 R_ACC_CART、R_ACC_POLE 置零，使目标函数与正动力学一致。控制器源码、配置和原日志未改动。

每组 2253 次求解；以下统计剔除每轮首次求解后的 2250 次。solve 是 opti.solve() 的墙钟耗时；完整调用还包括参数更新、热启动、结果提取和残差计算。

| 配置 | 平均 solve (ms) | 标准差 (ms) | P95 (ms) | 最大 (ms) | 完整调用平均 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward_original | 2.399 | 0.141 | 2.636 | 3.803 | 3.035 |
| inverse_original | 2.576 | 0.166 | 2.801 | 4.557 | 3.188 |
| forward_matched | 2.398 | 0.126 | 2.595 | 3.670 | 3.022 |
| inverse_matched | 2.567 | 0.131 | 2.800 | 3.693 | 3.171 |

9012 次求解全部成功；剔除首次后的 IPOPT 迭代次数全部为 4，solve 和完整调用均无超过 20 ms 的样本。首次求解（含求解器惰性初始化、不含控制器构建）为：

- forward_original: 17.986 ms, 17.107 ms, 17.156 ms
- inverse_original: 17.777 ms, 17.793 ms, 17.437 ms
- forward_matched: 16.879 ms, 17.475 ms, 17.424 ms
- inverse_matched: 16.724 ms, 17.070 ms, 16.921 ms

三轮 matched 回放的最大输出力差为 1.130e-14 N。

正动力学：104 个变量、167 个约束标量；逆动力学：144 个变量、207 个约束标量。原配置正动力学平均耗时降低约 6.9%；匹配目标后降低约 6.6%。这是当前 2 自由度倒立摆和 IPOPT 实现的结果，不能外推到之前的全身机器人与 Fatrop 实验。

已有 Webots 日志另行统计如下：按控制器 20 ms 调度选取新求解记录，避免每 10 ms 重复写入的计时被重复统计，并剔除首次。日志未记录运行环境，轨迹也不同，仅作历史参考。

| 旧 Webots 日志 | 样本数 | 平均 (ms) | P95 (ms) | 最大 (ms) |
| --- | ---: | ---: | ---: | ---: |
| forward | 2158 | 2.764 | 3.571 | 5.498 |
| inverse | 2153 | 3.063 | 4.186 | 17.767 |

复现（从仓库根目录运行，输出目录可另选）：

```bash
MPC_BENCHMARK_OUTPUT=mpc_speed_evaluation/retest_20260909 /home/tdt/anaconda3/envs/wewebot/bin/python mpc_speed_evaluation/benchmark.py
```

原始记录：raw_results.json；统计：summary.json；回放状态：states.json；历史 Webots 日志统计：existing_webots_logs.json。
