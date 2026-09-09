# Webots 单级倒立摆：Inverse-Dynamics MPC 最小复现

这个工程用一个 **2-DoF 单级倒立摆（cart-pole）** 验证论文《Whole-Body Inverse Dynamics MPC for Legged Loco-Manipulation》的核心算法结构。

这里不使用 Pinocchio、Fatrop 或 URDF。MPC 只依赖 **CasADi + IPOPT**；真实被控对象由 **Webots/ODE** 负责动力学仿真。

## 1. 工程结构

```text
webots_inverse_dynamics_mpc/
├── README.md
├── requirements.txt
├── worlds/
│   └── cartpole_inverse_dynamics_mpc.wbt
├── controllers/
│   └── inverse_dynamics_mpc/
│       ├── config.py
│       ├── dynamics.py
│       ├── mpc.py
│       └── inverse_dynamics_mpc.py
├── tools/
│   ├── offline_validate.py
│   └── plot_log.py
└── logs/
```

## 2. 和论文算法的对应关系

单级倒立摆状态定义为

\[
q = \begin{bmatrix}x\\\theta\end{bmatrix},\qquad
v = \begin{bmatrix}\dot x\\\dot\theta\end{bmatrix}.
\]

其中：

- `x`：小车位置；
- `theta`：摆杆角度，`theta=0` 为竖直向上；
- 小车由水平力 `F` 驱动；
- 摆杆铰链没有电机，是欠驱动自由度。

### 2.1 运动学状态传播

和论文公式 (4) 一样，MPC 用显式 Euler：

\[
q_{k+1}=q_k+v_k\Delta t_k,
\]

\[
v_{k+1}=v_k+a_k\Delta t_k.
\]

注意：这里的状态传播**不调用 forward dynamics**。

### 2.2 Inverse dynamics path constraint

单级倒立摆完整刚体动力学为

\[
M(q)a+h(q,v)=
\begin{bmatrix}
F\\0
\end{bmatrix}.
\]

MPC 把 `a_k` 和 `F_k` 都作为优化变量，然后计算

\[
\tau_{id}=M(q_k)a_k+h(q_k,v_k).
\]

并施加

\[
\tau_{id,0}=F_k,
\]

\[
\tau_{id,1}=0.
\]

第二行等于零的原因是：摆杆铰链没有电机。

这和论文中 floating-base 机器人要求 RNEA 的未驱动基座行等于零是同一个思想，只是这里欠驱动自由度只有一个。

### 2.3 MPC 决策变量

每个预测节点优化

\[
q_k,\;v_k,\;a_k,\;F_k.
\]

所以算法数据流是：

```text
q_k, v_k, a_k
      │
      ├── explicit Euler ──> q_{k+1}, v_{k+1}
      │
      └── inverse dynamics: M(q)a+h
                    │
                    ├── actuated row = F_k
                    └── passive row  = 0
```

这就是要验证的核心。

## 3. 动力学模型

Webots 中摆杆绕 `+y` 轴旋转，正角度向 `+x` 倾斜。设：

- 小车质量 `M`；
- 摆杆质量 `m`；
- 摆杆质心到转轴距离 `l`；
- 摆杆关于转轴的惯量 `J`。

质量矩阵：

\[
M(q)=
\begin{bmatrix}
M+m & ml\cos\theta\\
ml\cos\theta & J
\end{bmatrix}.
\]

偏置项：

\[
h(q,v)=
\begin{bmatrix}
-ml\sin\theta\dot\theta^2\\
-mgl\sin\theta
\end{bmatrix}.
\]

因此 inverse dynamics：

\[
\begin{bmatrix}
F_{required}\\
\tau_{hinge,required}
\end{bmatrix}
=
M(q)a+h(q,v).
\]

MPC 强制

\[
F_{required}=F_k,
\qquad
\tau_{hinge,required}=0.
\]

## 4. 安装依赖

建议让 Webots 使用你安装了 CasADi 的同一个 Python。

```bash
cd webots_inverse_dynamics_mpc
python3 -m pip install -r requirements.txt
```

验证：

```bash
python3 -c "import casadi, numpy; print(casadi.__version__)"
```

如果 Webots Controller 控制台报 `No module named casadi`，在 Webots 的 Python command 设置里选择上面同一个 `python3` 解释器。

## 5. 先做纯 Python 离线验证

在打开 Webots 之前先运行：

```bash
python3 tools/offline_validate.py
```

这里：

- **MPC 内部仍然只使用 inverse dynamics**；
- `tools/offline_validate.py` 仅用 forward dynamics 充当“外部物理世界”；
- 在 Webots 中，这个“外部物理世界”会被 Webots/ODE 替代。

运行成功时会输出：

- 最终状态；
- MPC 平均求解时间；
- inverse-dynamics 被动铰链约束残差；
- `PASS`。

数据保存到：

```text
logs/offline_validation.csv
```

## 6. 在 Webots 中运行

1. 启动 Webots。
2. 打开：

```text
worlds/cartpole_inverse_dynamics_mpc.wbt
```

3. 点击 Run。

世界使用 Webots 内置背景、光源和地板节点，不需要下载外部 PROTO。
`WorldInfo.gravity` 必须写成标量 `9.81`；`coordinateSystem "ENU"` 下重力沿 -Z，
不要写成 `gravity 0 0 -9.81`，这种向量写法会导致世界解析失败。

初始摆角是：

\[
\theta_0=0.08\;rad\approx4.58^\circ.
\]

MPC 应该驱动小车把摆杆重新拉回：

\[
\theta\rightarrow0,
\qquad
x\rightarrow0.
\]

控制器直接调用：

```python
cart_motor.setForce(F)
```

因此 Webots 自带的位置 PID 没有参与控制；发送的是 MPC 求出来的真实水平力。

运行日志保存到：

```text
logs/webots_log.csv
```

## 7. 查看日志

```bash
python3 tools/plot_log.py logs/webots_log.csv
```

会显示：

- 摆角；
- 小车位置；
- MPC 水平控制力。

控制器日志中还保存了：

```text
passive_hinge_id_residual_Nm
```

它对应

\[
\tau_{hinge,required}=0.
\]

理想情况下应接近数值零。这是最直接的 inverse-dynamics 约束验证量之一。

## 8. 最重要的代码

核心代码在：

```text
controllers/inverse_dynamics_mpc/mpc.py
```

特别看这几行：

```python
tau_id = inverse_dynamics_ca(qk, vk, ak)
opti.subject_to(tau_id[0] == uk)
opti.subject_to(tau_id[1] == 0.0)
```

它们就是整个 inverse-dynamics MPC 的核心。

然后再看：

```python
q_next = qk + vk * dt
v_next = vk + ak * dt
opti.subject_to(X[:, k + 1] == vertcat(q_next, v_next))
```

你会看到：

- `a` 是优化变量；
- `a` 直接用于状态传播；
- 但 `a` 不能随便取，因为 inverse-dynamics equality constraint 会约束它；
- Webots 最终只接收优化得到的 `F`。

## 9. Webots 与解析模型参数必须一致

如果你修改 `.wbt` 里的以下参数：

- cart mass；
- pole mass；
- pole length；
- pole width；
- gravity；

必须同步修改：

```text
controllers/inverse_dynamics_mpc/config.py
```

否则你故意引入了 model mismatch。

这也可以作为后续实验：逐步修改 Webots 真实质量但不修改 MPC 模型，观察 inverse-dynamics MPC 对模型误差的敏感性。
