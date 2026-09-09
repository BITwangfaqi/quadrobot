# wb_mpc_locoman → Webots / quadruped_arm

本工程直接移植仓库根目录的 `wb_mpc_locoman` 算法源码，以
`Quadruped_MPC/worlds/quadruped_arm.wbt` 的机器人为控制对象。

完整默认模型已编译并接入默认启动入口。当前已删除 IPOPT 自动回退：
每次 MPC 调度只调用配置的求解器一次，不合格解立即进入位置保持。
机械臂启动/参考姿态已移入限位内部，并已为新参考重新生成、编译和部署共享库。
默认编译版通过 30 秒平地小跑测试：351 次 Fatrop 调用全部成功，零回退、零故障。
完整调用平均 91.3 ms、p95 111.0 ms，307/351 次超过 80 ms，尚未达到墙钟实时。
6 秒站立/机械臂末端上升测试也通过，50 次求解全部成功。
详情见 `validation/RESULTS.md`。
没有复制、导入或调用 `inv_dyn_mpc` 的控制器、模型、优化器、步态或编译代码。

`wb_mpc/` 保留原作者的五种 OCP、动力学函数、步态和求解器实现；
新增适配层负责从 WBT 建模、读取实际状态、验收解、执行调度周期内的预测轨迹。
保留原作者 MIT 许可证，来源文件的 SHA-256 在 `upstream_sha256.json`，
原算法目录的全部差异在 `UPSTREAM_CHANGES.patch`。

**这是同步在线物理闭环，不是离线轨迹回放，也不保证墙钟实时。**
求解期间 Webots 等待控制器；日志分别记录仿真时间与完整调用墙钟耗时。
`deadline_misses` 表示超过 MPC 调度周期的次数。

## 启动

本机使用已经验证的解释器：

```bash
/home/tdt/anaconda3/envs/wb-mpc/bin/python \
  /home/tdt/quadrobot/Quadruped_MPC/controllers/wb_mpc_webots/main.py
```

先在 Webots 打开原来的 `quadruped_arm.wbt`，保持 Robot 的
`controller "<extern>"`，启动仿真等待连接，再执行上面命令。
同一个机器人只连接一个外部控制器。原世界的 `basicTimeStep` 已按要求改为 10 ms，Robot 的外部控制器设置保留。
更新后先停止旧控制器、重置仿真，再启动本入口；等位置启动和 OCP 初始化完成后按 U 切小跑。
VS Code 运行 `main.py` 时也选择 `wb-mpc` 解释器。

不传 `--config` 时读取本目录 `deployment.json`，加载
`.cache/deployed/solver_function.so` 中的完整 Fatrop 求解器、轨迹解码和约束检查。
终端会打印 `backend: .../solver_function.so`。显式传
`--config /home/tdt/quadrobot/Quadruped_MPC/controllers/wb_mpc_webots/default.json`
可运行相同参数的解释执行版本。配置中的相对共享库路径相对于配置文件目录解析。

也可以手动把 Robot.controller 改为 `wb_mpc_webots`，重新加载世界后直接运行；
Webots 会使用同名 `wb_mpc_webots.py` 和本目录的 `runtime.ini`。
`runtime.ini` 使用本机解释器路径，换机器需要修改。

其他机器可在本目录创建依赖环境：

```bash
conda env create -f environment.yaml
conda activate wb-mpc
python export_solver.py --config default.json --nominal deployment_nominal.json \
  --output .cache/deployed --compile --optimization 0
python main.py
```

默认 Webots 安装位置为 `/usr/local/webots`，可通过 `WEBOTS_HOME` 修改。
世界 `basicTimeStep` 和反馈周期默认均为 10 ms；控制器自动读取世界步长，支持 1、2、10 ms。
MPC 默认周期为 80 ms（12.5 Hz），每 8 个 10 ms 仿真步求解一次。
中间 7 步复用上一条预测，仍每 10 ms 更新测量、关节反馈和力矩输出。
首次进入控制或切换步态时立即求解一次，再按 80 ms 调度。
这是仿真时间调度；同步求解仍会暂停仿真推进，80 ms 墙钟耗时不等于仿真自动走了 8 步。

启动后先用位置模式保持 2 秒：腿维持 WBT 编码器零位，机械臂肩肘使用
`joint2=0.15 rad、joint3=-0.30 rad`，其余机械臂关节保持零位。
WBT 肩肘零位恰好在硬限位上，轻微向外运动就可能使第一步位置预测违反关节约束。
解释执行版本以启动后实际静止姿态作为参考；编译版本的启动目标和参考都使用
导出时记录的已落地姿态（`deployment_nominal.json`），
构建 OCP，进入站立力矩控制。初始化构建只执行一次，之后更新参数求解。

| 操作 | 按键 |
| --- | --- |
| 站立 / 小跑 / 行走 | J / U / O |
| 前后 / 左右平移速度 | 按住 W/S / A/D，需处于移动步态 |
| 转向速度 | 按住 Q/E |
| 机械臂末端向上 / 向下 | 按住 I/K |
| 机械臂末端前后 / 左右 | 按住方向键 |
| 锁存位置保持 | 空格；J/U/O 恢复 |

机械臂按键设置速度目标，释放后目标速度归零；不是绝对位置锁定任务。
机器人已接触地面后才开始 MPC，正常启动不需要手动摆成 B2 的姿态。

## 文件结构与来源

```text
wb_mpc_webots/
  main.py, wb_mpc_webots.py   Webots 入口、测量、执行和调度
  world_model.py             独立 WBT 解析器与 Pinocchio 模型适配
  mpc_bridge.py              调用原版 make_ocp / solver_function，解码和验收
  wb_mpc/
    args.py                 原版模型与求解器选项
    dynamics/               原版全身/质心动力学
    optimization/           原版 OCP 基类及五种子类
    utils/                  原版步态、机器人辅助、可视化和调试代码
  offline.py                WBT 模型上的原版预测自推进模式，可保存/绘图
  export_solver.py          Fatrop C 导出、编译、等价性检查
  validate.py               模型和离线求解检查
  check_webots.py           独立 Webots 物理闭环测试
  reference/                原入口、README、环境和 codegen 说明的快照
  validation/               验证结果；大型运行日志在忽略的 runs/ 中
```

`reference/main.py` 是来源快照，不是新工程运行入口。
原版 B2/Z1 加载辅助保留用于对照；不重复复制无关的约 384 MB B2/Z1 网格资产。
新工程使用 WBT 中的网格和惯性数据，不依赖根目录原算法包或其他控制器。

原算法文件只做以下调整：

1. 将绝对模块导入改成包内相对导入，避免和其他工程的 `utils/dynamics` 冲突。
2. 公共 OCP 取机械臂初始构型时，对 `centroidal_vel` 使用其实际 `[h/m,q]` 状态布局。
3. OSQP 返回无效状态或空解时抛错，避免把结果下发给执行器。

动力学方程、权重、非均匀 Euler 网格、RNEA 近端力矩节点、摩擦锥、
支撑足零速度、摆动足竖直样条、末端速度/力等式以及原版 warm start 均保留。
没有改成旧控制器的 NLP 或单独的 WBC QP。

## Webots 平台适配

WBT 中的显式 Robot/Solid/HingeJoint/SliderJoint 树直接构造成 Pinocchio 模型。
质量、COM、惯量、轴、anchor、endpoint 零位姿态、限位均来自原 WBT。
解析器不是通用 PROTO 展开器：依赖当前显式 Bigdog Robot，非零关节初始
`position` 会明确报错；换成其他结构需要补充相应解析规则。

| 项目 | 当前映射 |
| --- | --- |
| 总质量 | 18.134388 kg，包含固定机械臂末端和夹爪惯量 |
| 默认维度 | nq=23、nv=22、nj=16、nf=15 |
| 关节顺序 | FL、FR、BL、BR 每腿 hip/leg/foot，然后 joint1～joint4 |
| 接触顺序 | 原版 FR、FL、RR、RL；对应 Webots FR、FL、BR、BL |
| 锁定关节 | joint5、joint6、joint7、joint8；后两者为夹爪 slider |
| 浮基 frame | `base_link` 表示机器人根刚体，WBT 同名机械臂座另记为 `wbt/base_link` |
| 足端 frame | 各 `*_4` Solid 球形脚的中心，沿用原版点接触运动学 |
| 工具 frame | gripper_base 局部 `[0,0,0.1358]`，对应两夹爪关节原点中心 |
| 状态来源 | GPS 位置/世界线速度，IMU 姿态，gyro，编码器差分低通速度 |
| 速度坐标 | GPS 线速度旋转到基座局部；gyro 按当前 WBT 的根部无旋转安装读取 |

WBT 原始零位已经是弯腿姿态，不能套用 B2 SRDF 的关节角。
固定腕部和夹爪在模型中是刚性约束，在 Webots 中用位置控制保持，因此存在执行误差。
改变传感器安装位置/方向时，必须同步修改 `Interface.measure()`。

适配层默认设置与原示例的区别：

- 初始任务为站立、速度和末端力为零；待启动稳定后手动切步态。
- 默认 MPC 仍每 80 ms 求解一次，预测网格为 20～40 ms，14 节点。
  执行层解码覆盖 80 ms 的全部节点，各 10 ms 步按当前节点更新预测参考。
  RNEA 力矩约束节点自动扩展到覆盖执行窗口，避免执行无力矩约束的远端节点。
- 腿部位置/速度反馈分别为 `[60,60,20]` / `[0.5,0.5,0.1]`，按 hip/leg/foot 重复四次。
  机械臂肩肘为 300/12，腕部为 5/0.04。膝、腕轻惯量关节不能沿用 2 ms 下的阻尼。
- 腿和机械臂位置参考跨求解周期连续积分，并裁剪到关节限位及相对测量的偏差上限。
- `gait_adapter.py` 在原版小跑接触表上引入每半周期 160 ms 的四足支撑过渡，
  默认步态周期 1.12 s，摆脚高度 0.025 m，起落脚速度为 0.02/-0.04 m/s。
  这是 Webots 平台适配，与原版无过渡小跑不同；原版步态源码仍保留。
- 每个反馈步使用实测状态、预测加速度与接触力重新计算逆动力学力矩。
  未触地时去掉对应预测支撑反力项；机身位置/姿态反馈经小型最小二乘分配到
  实际触地且预测承重的脚。最终力矩经关节反馈叠加及电机限幅。
  这些是新增执行层计算，不是再次运行 MPC，也没有调用旧工程的 WBC。
- Fatrop/Ipopt 的求解容差为 1e-5，执行验收仍为 1e-3。默认只构建和调用 Fatrop，
  返回不合格解时直接进入保持，不再初始化或调用备用 IPOPT 求解器。
  已删除 `ipopt_fallback` 配置项；显式选择 `solver=ipopt` 的独立求解模式仍保留。
- 单次解不合格立即保持；连续三次失败或机身/关节运动异常锁存保持，J 恢复站立。
  已经跌倒时应重置世界，不应直接恢复。锁存保持不是小跑成功，验收会判为失败。
- 世界默认接触反弹系数设为 0；隔离物理测试复制原世界的完整 WorldInfo。
- 源码中用于离线绘图的历史列表在实时适配器中仅保留最近两项，防止内存持续增长。
- GUI 用 Webots 自身的实时画面；不启动原示例末尾的 Meshcat 50 次离线回放。

## 配置模型与求解器

```bash
python main.py --config default.json --duration 10 --log .cache/run.jsonl
python main.py --config original_parameters.json
python offline.py --steps 200 --gait stand --plot
```

JSON 字段对应 `mpc_bridge.Settings`。支持原版五种 `dynamics`：
`whole_body_rnea`、`whole_body_aba`、`whole_body_acc`、`centroidal_acc`、`centroidal_vel`。
三个 acc/centroidal 模型可配置 `include_base`；RNEA 可配置 `include_acc`，
但原版 Fatrop 自动结构检测要求 `include_acc=true`。

求解器支持 `fatrop`、`ipopt`、`osqp`。OSQP 使用原版 SQP，默认 2 轮、每个
QP 20 次迭代；本机站立样本 CV 约 0.094，无法通过 0.001 的执行验收，
因此默认配置下只适合算法对照，不能视为已验证可运行的物理控制配置。
可通过 `sqp_iters`、`qp_max_iter` 增加预算进行研究。

`arm_force` 是三维世界系外力，默认 `[0,0,0]`。非零目标要求场景真实存在
对应接触/连接；该参数不会自动在 Webots 创建外力。

保留原版模型的限制：固定接触表；无地形、碰撞、不穿透、冲击约束；
非 RNEA/ABA 分支的近端力矩约束为原版静态估计。适配器额外用完整 RNEA
验收实际要下发的近期力矩。触地传感器参与执行力矩修正，不直接修改 MPC 的接触计划。
末端速度目标的坐标变换仍沿用原作者实现，大角度转向/姿态变化未做完整验证。

## C 导出与编译

```bash
python export_solver.py --config default.json --nominal deployment_nominal.json \
  --output .cache/deployed --compile --optimization 0
```

在本目录、`wb-mpc` 环境执行上述命令。导出原版 `solver_function`、完整状态/参数到
覆盖执行周期的节点的 `webots_decode`，以及约束函数 `g_data`，一起编译进共享库。
编译需要 `cc`、Fatrop/Blasfeo 头文件和共享库，默认从当前 Conda 环境的
include/lib 查找；`CC` 可指定编译器。完整默认 14 节点问题的三个 C 文件共约 68 MiB。
本次部署使用 GCC `-O0`，新参考的完整编译约 138 秒，共享库约 71 MiB。
`-O1`/`-O3` 在本机处理巨大生成函数时编译耗时过长，已停止，未作为成功构建部署。
需要尝试其他优化等级时，使用独立 `--output` 目录；可先复制三个 C 文件和元数据，再用
`--reuse-generated` 省去重复生成。高等级编译是否进一步提速，需要完成后单独实测。
这里只编译数值计算；传感器读取、热启动参数更新、轨迹重构和电机下发仍由 Python 执行。
IPOPT 自动回退已删除。机械臂参考姿态已移入关节限位内部，部署库需与该参考一致。
旧构建元数据中的 `ipopt_fallback` 仅作为历史字段忽略，不会启用回退；
其他配置、模型和库校验保持生效。

输出 `solver_function.so` 及同名 `.json` 元数据。`deployment.json` 已指定加载路径，
其余 Settings 必须和导出时一致。
启动会使用导出时记录的参考姿态，实际初始状态仍来自传感器；同时校验
世界内容、机器人自由度、参考姿态、Motor 限位、配置、CasADi 版本、数值等价性标记和
库的 SHA-256，拒绝错配或未验证的库。修改动力学代码、世界或优化配置后需重新导出编译。
`.cache` 不纳入版本控制；换机器/环境后也需要重建，不能直接复制带本机依赖路径的库。
若要运行其他自由度或模型，先显式选择对应解释执行配置，再为该配置单独导出。

用同一段测量数据、同一组参数和热启动进行对照测试：

```bash
python benchmark_compiled.py --trace validation/runs/<测试目录>/telemetry.jsonl
```

该测试分别计时两种 Fatrop 实现，不包括 Python 更新，不使用 IPOPT 回退；
真实闭环的完整耗时以 `check_webots.py` 输出为准，不能把两者混为一谈。

## 验证与范围

```bash
python validate.py --solve --all-models
python validate.py --solve --solver ipopt --output validation/ipopt.json
python check_webots.py --duration 5
python check_webots.py --duration 6 --scenario trot
python check_webots.py --duration 5 --scenario arm
python check_webots.py --duration 6 --scenario trot --velocity 0.03
```

离线检查覆盖来源校验、WBT 独立刚体树变换/质心、重力有限差分、五模型求解、
符号与数值 RNEA、外力符号、输出形状和轨迹过期拒绝。

物理检查复制原 WBT 的**完整 Robot 节点**（保留实际惯性、碰撞、电机和传感器），
只在隔离的测试世界中改成内部 Supervisor 测试入口，并换用显式平地，避免远程 PROTO。
不会修改用户原世界或连接正在运行的外部控制器。
测试记录高度、姿态、力矩模式占比、求解失败/超时；小跑还检查实际对角触地交替
与腿运动，机械臂检查编码器重建与 Supervisor 直接测量的末端位移，
并检查两者的高度差；前进测试检查基座位移。Supervisor 仅用于验收，
控制器反馈不使用其真值。每次新测试保存配置和入口源码哈希。

结果说明见 `validation/RESULTS.md`。这些短时平地测试不覆盖原世界中的石块、
窄桥、所有键盘组合、长时间运行或真机。原世界仍可以直接使用新控制器，
但原算法的平地接触假设不等于具备越障能力。
