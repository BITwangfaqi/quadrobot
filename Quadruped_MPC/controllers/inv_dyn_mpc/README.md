# 四足 + 机械臂全身逆动力学 MPC

本控制器面向 `Quadruped_MPC/worlds/quadruped_arm.wbt`，复现论文 [Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation](https://doi.org/10.1109/LRA.2025.3636005) 的全身非线性逆动力学优化结构。对照了作者[公开实现](https://github.com/lukasmolnar/wb-mpc-locoman/tree/80e906d35d91783e85e1ef994023ca9082dc40c3) 的 `ocp_whole_body_rnea.py`、`ocp.py`、参数及 C 编译入口。

**80 Hz 是求解性能目标，并非已验证的实时保证。** 求解仍同步执行，超时会使 Webots 慢于实时。控制台的 `deadline_misses` 统计完整求解与输出验收超过 12.5 ms 的次数，包含冷启动和失败。不能把仿真时间中的调度频率当作墙钟求解频率。

## 启动方式

保持与 `my_BIGDOG4_MPC.py` 相同：Webots 设置 `controller "<extern>"`，在 VSCode 运行 Python 脚本。

1. 在 Webots 重新加载 `quadruped_arm.wbt`，确认 `WorldInfo.basicTimeStep = 2`（毫秒）。启动仿真，等待外部控制器连接。
2. VSCode 选择解释器 `/home/tdt/anaconda3/envs/wewebot/bin/python`。可使用项目 `.vscode/launch.json` 的 `inv_dyn_mpc (Webots extern, wewebot)` 配置，也可打开 `inv_dyn_mpc.py` 后运行文件。
3. 终端启动命令：

```bash
/home/tdt/anaconda3/envs/wewebot/bin/python /home/tdt/quadrobot/Quadruped_MPC/controllers/inv_dyn_mpc/inv_dyn_mpc.py
```

脚本自动查找 `/usr/local/webots/lib/controller/python`。其他安装路径通过 `WEBOTS_HOME` 指定。连接同一机器人时只运行一个控制器。无需修改 Webots 的 Python command；解释器由 VSCode 决定。

首次运行会生成并编译**完整 Fatrop 求解入口、精确导数及轨迹恢复函数**，本机约需 1–2 分钟。后续复用 `.cache/<hash>/solver.so`。模型、优化代码、编译配置或求解参数变化会产生新的缓存；打开的旧 `oracle.c` 不代表当前运行代码。构建期间机器人保持位置控制，仿真等待控制器。

本机 `wewebot` 提供 NumPy 1.26.4、CasADi 3.7.0、Fatrop/BLASFEO 及编译头文件。其他机器仅安装 `requirements.txt` 可运行 `jit=false` 诊断路径；完整 C 编译还需要 clang 或 gcc、Fatrop 和 BLASFEO 的头文件/共享库。IPOPT 用作诊断，未实现完整 IPOPT C 编译。

## 10 ms 下的失稳

论文中 MPC 为 80 Hz，关节插值与反馈为 500 Hz，两者是不同周期。本控制器使用 2 ms 物理/反馈步长，MPC 调度周期为 12.5 ms；在 2 ms 网格上保留累积截止时间，以 12/14 ms 间隔交替调度，避免每次取整导致降为约 71 Hz。

旧代码在 10 ms 下站立可能正常，但切换小跑后摆动腿的采样力矩反馈失稳。对当前 WBT 质量矩阵、对角支撑约束及 PD 增益线性化，2 ms 的离散谱半径约为 1（含中性模态），10 ms 为 6.332891，大于 1。实际 Webots 对照中，旧代码 10 ms 在 5 s 切换小跑后约 5.44 s 失稳；2 ms 对照未出现同样的快速发散。

因此世界文件已恢复 2 ms。脚本检测到大于 2 ms 的物理步长会拒绝启动并提示重新加载世界，避免继续施加不稳定反馈。不要为了匹配 MPC 频率把 `basicTimeStep` 改回 10 ms。

## 与论文对应的实现

| 项目 | 当前实现 |
| --- | --- |
| 刚体模型 | 从 WBT 解析浮基、腿、机械臂、固定夹爪的质量、质心、惯量和关节坐标；总质量 18.134388 kg |
| 自由度 | 默认锁定机械臂最后两关节，与论文的 12 腿 + 4 臂 + 6 浮基一致：22 个速度自由度；锁定部分惯量保留 |
| 状态 | 初始构型切空间位移和广义速度；初始实测状态直接代入，等价消去固定变量与等式 |
| 输入 | 广义加速度、四足接触力；k=0,1,2 另有显式关节力矩 |
| 动力学 | 全身 RNEA 作为路径等式；前 3 节点保留全部动力学和力矩边界，后段消去力矩、保留 6 行浮基方程 |
| 积分 | 切空间显式 Euler；速度显式 Euler；SE(3) 构型恢复 |
| 网格 | 14 个区间、15 个状态节点，首步 0.01 s，增长比 1.193776641714434，末步 0.1 s，总时域约 0.56445 s |
| 足端 | 固定接触表，支撑足零线速度；摆动足零接触力、跟踪竖直三次曲线速度，水平速度自由 |
| 工具 | 未来路径节点的末端线速度硬约束；位置误差在测量状态上转换为本次求解的速度指令 |
| 代价 | 初始切空间二次状态误差、加速度、足端力及前 3 节点关节力矩；默认权重对照作者实现 |
| 求解 | Fatrop 内点法、阶段稀疏结构、精确 Jacobian 和拉格朗日 Hessian、完整 C 入口及 `-O3 -march=native` 编译 |
| 执行 | 首两区间的关节位置/速度预测及力矩线性插值，加关节 PD；无单独 WBC QP |

当前 `wewebot` 的 Pinocchio/EigenPy 数值接口存在 NumPy ABI 不兼容。模型加载器可通过本机工作正常的 `/usr/bin/python3` 和 `/opt/openrobots/lib/python3.10/site-packages` 导出原生 `pinocchio.casadi` 函数，并在 `wewebot` 中加载。跨环境仅交换序列化函数，不混用两个解释器的 Pinocchio 动态库。导出结果与独立刚体树 RNEA 比较后才使用。其他机器无该环境时自动使用 CasADi 刚体树后端，速度应重新测量。可用 `INV_DYN_PIN_PYTHON`、`INV_DYN_PIN_PATH` 指定导出环境。

四足接触相位复用了 `my_BIGDOG4_MPC/getMpcTable.py`。旧控制器的单刚体近似、关节角偏置和力矩符号不适用于这个全身模型，没有直接套用。WBT 原始编码器零位已经对应弯腿姿态。

以下是明确的平台适配，不能称为论文所有实验的完全一致复现：

- 机器人为当前四足 + Piper，并非论文 B2 + Z1；电机边界来自 WBT 和实时设备两者中更严格的一方。腕部与夹爪在 Webots 中通过位置控制保持，非理想机械锁止。
- 摩擦系数为 0.6，足端力有 400 N 分量边界；支撑力参考均匀分配，未沿用 B2/Z1 的前后载荷比例。
- 保留终端关节位置/速度边界；作者代码终端只有代价。首个不可由当前加速度修正的位置节点允许 0.002 rad 测量容差，后续节点严格限位。
- 工具指定外力作为参数消元；默认自由空间外力为零。接触推拉实验需要世界中实际存在对应接触或连接。
- 最大迭代数为 80。作者示例的 10 次上限在本模型的实际 Webots 接触切换中不足，因此未强行使用。必须正常收敛且通过有限值、约束残差检查才下发解。
- 步态固定，触地传感器用于诊断，尚未实现地形估计或接触自适应。已移除旧实现额外的足端高度约束和高度稳定化代价。
- 热启动复用上一解；接触表改变时重置足端力。默认单线程导数；本机测试未发现 OpenMP 有稳定收益。`derivative_threads>1` 为可选编译路径，需要相应 OpenMP 开发库。

## 操作及配置

默认平滑位置启动 3 s，稳定 1 s，随后站立力矩控制。

| 按键 | 操作 |
| --- | --- |
| U / J | 小跑 / 站立或恢复 |
| W / S、A / D | 小跑时前后、左右运动 |
| Q / E | 小跑时转向 |
| I / K | 工具上移 / 下移 |
| 方向键 | 工具前后、左右运动 |
| 空格 | 位置保持 |

`--config <file.json>` 覆盖 `config.py` 中的字段；`paper_grid.json` 与默认网格一致。`--duration 10` 表示运行到仿真时间累计 10 s 后位置保持退出，包含启动阶段。

只允许从前两个受力矩约束的区间采样命令，默认最长约 21.94 ms（仿真时间）。求解失败后旧解超过此窗口会进入位置保持并继续重试。该窗口不会因把 `solution_timeout` 设大而允许力矩外推。同步仿真中物理时间在求解期间等待，因此这不代表已实现硬件上的异步时延补偿。

## 验证

```bash
cd /home/tdt/quadrobot/Quadruped_MPC/controllers/inv_dyn_mpc
OPENBLAS_NUM_THREADS=1 /home/tdt/anaconda3/envs/wewebot/bin/python self_check.py --solve --steps 5
OPENBLAS_NUM_THREADS=1 /home/tdt/anaconda3/envs/wewebot/bin/python self_check.py --solve --tasks --no-jit --steps 2
OPENBLAS_NUM_THREADS=1 /home/tdt/anaconda3/envs/wewebot/bin/python benchmark.py --samples 100 --mode stand --output validation/stand.json
OPENBLAS_NUM_THREADS=1 /home/tdt/anaconda3/envs/wewebot/bin/python benchmark.py --samples 100 --mode trot --output validation/trot.json
OPENBLAS_NUM_THREADS=1 /home/tdt/anaconda3/envs/wewebot/bin/python benchmark.py --samples 100 --mode tool --output validation/tool.json
```

自检覆盖独立模型交叉验证、SE(3) 零位导数、接触力符号、采样反馈失稳诊断、有限差分 Jacobian/Hessian、编译入口返回值，以及独立受约束正向动力学短时站立。默认自检网格与控制器一致；小规模诊断可加 `--horizon 4 --no-jit`。

`benchmark.py` 包含参数构造、热启动、求解、验收和轨迹恢复，记录均值、P95、P99、最大耗时、失败和 12.5 ms 超时。冷启动单列且仍参与全部样本的验收；失败不被过滤。其状态跟随优化器预测，**只用于求解性能测量，不是独立物理闭环验证**。

实际物理验证使用 `webots_check.py`，在独立临时项目复制同一机器人和显式平地，不改动用户运行中的 Webots：

```bash
/home/tdt/anaconda3/envs/wewebot/bin/python webots_check.py --scenario trot --duration 10 --timestep 2 --output /tmp/inv_dyn_webots_check --xvfb /tmp/inv_dyn_xvfb/usr/bin/Xvfb
```

需要 Webots 和 Xvfb；其他机器需指定其本机 Xvfb 路径。记录 `states.jsonl`、每次完整求解的 `solves.jsonl`、控制器日志。自动化小跑在仿真 5 s 后按 U+W；工具测试按 I。此平地测试不包含论文的外力扰动、推拉重物或擦拭实验，也不能证明所有键盘指令组合稳定。
