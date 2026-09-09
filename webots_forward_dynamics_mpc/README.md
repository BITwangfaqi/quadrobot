# Webots 单级倒立摆：Forward-Dynamics MPC 最小复现

这个工程是在前一个 `Inverse-Dynamics MPC` 倒立摆工程基础上做的**一一对应正动力学版本**。机器人、Webots 世界、状态定义、预测时域、代价权重、执行频率都尽量保持一致，唯一真正改变的是 MPC 的动力学建模方式。

本工程不用 Pinocchio、Fatrop 或 URDF。MPC 使用 **CasADi + IPOPT**；真实被控对象由 **Webots/ODE** 负责仿真。

## 1. 工程结构

```text
webots_forward_dynamics_mpc/
├── README.md
├── requirements.txt
├── worlds/
│   └── cartpole_forward_dynamics_mpc.wbt
├── controllers/
│   └── forward_dynamics_mpc/
│       ├── config.py
│       ├── dynamics.py
│       ├── mpc.py
│       └── forward_dynamics_mpc.py
├── tools/
│   ├── offline_validate.py
│   └── plot_log.py
└── logs/
```

## 2. 状态与控制输入

定义

\[
q=\begin{bmatrix}x\\\theta\end{bmatrix},\qquad
v=\begin{bmatrix}\dot x\\\dot\theta\end{bmatrix},
\]

完整状态为

\[
X=\begin{bmatrix}x&\theta&\dot x&\dot\theta\end{bmatrix}^T.
\]

控制输入只有小车水平力

\[
u=F.
\]

摆杆铰链是被动关节，因此广义输入为

\[
\tau=\begin{bmatrix}F\\0\end{bmatrix}.
\]

## 3. 正动力学方程

刚体动力学仍然是

\[
M(q)a+h(q,v)=\begin{bmatrix}F\\0\end{bmatrix}.
\]

但是本工程**不把加速度 `a` 当成优化变量**，而是直接求

\[
\boxed{
a=M(q)^{-1}\left(\begin{bmatrix}F\\0\end{bmatrix}-h(q,v)\right)
}
\]

在数学上这是一个线性求解。为了让这个 2×2 教学模型可以被 CasADi 完整 `expand`，代码把 2×2 线性方程的解析解直接写出来，而不是显式计算矩阵逆。对复杂机器人而言，这一步对应 `ABA / forward dynamics`。

## 4. MPC 状态转移

每个预测节点只优化：

\[
X_k,\;F_k.
\]

先通过正动力学计算：

\[
a_k=f_{FD}(q_k,v_k,F_k),
\]

然后用显式 Euler：

\[
q_{k+1}=q_k+v_k\Delta t_k,
\]

\[
v_{k+1}=v_k+a_k\Delta t_k.
\]

因此动力学直接嵌入状态转移函数：

```text
X_k, F_k
   │
   ▼
Forward Dynamics
M(q)^-1([F,0]-h)
   │
   ▼
   a_k
   │
   ▼
Euler integration
   │
   ▼
X_{k+1}
```

核心代码：

```python
x_next = state_transition_ca(xk, uk, dt)
opti.subject_to(X[:, k + 1] == x_next)
```

## 5. 和前一个 Inverse-Dynamics MPC 的根本区别

### Inverse dynamics 版本

决策变量：

\[
X_k,\;a_k,\;F_k.
\]

状态传播：

\[
q_{k+1}=q_k+v_kdt,
\qquad
v_{k+1}=v_k+a_kdt.
\]

再额外要求：

\[
M(q_k)a_k+h(q_k,v_k)=\begin{bmatrix}F_k\\0\end{bmatrix}.
\]

也就是：**动力学作为 path constraint**。

### Forward dynamics 版本

决策变量只有：

\[
X_k,\;F_k.
\]

加速度不是独立变量：

\[
a_k=M^{-1}(\tau_k-h_k).
\]

然后直接：

\[
X_{k+1}=f(X_k,F_k).
\]

也就是：**动力学嵌入 state transition**。

这是你用倒立摆验证两种 MPC 最值得观察的区别。

## 6. 倒立摆具体动力学

设：

- 小车质量 `M`；
- 摆杆质量 `m`；
- 摆杆质心到转轴距离 `l`；
- 摆杆关于转轴的惯量 `J`；
- `theta=0` 为竖直向上。

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

于是：

\[
\begin{bmatrix}
\ddot x\\
\ddot\theta
\end{bmatrix}
=
M(q)^{-1}
\left(
\begin{bmatrix}
F\\0
\end{bmatrix}
-h(q,v)
\right).
\]

## 7. 代价函数

MPC 最小化：

\[
J=\sum_{k=0}^{N-1}
\left(
Q_x x_k^2+Q_\theta\theta_k^2
+Q_{\dot x}\dot x_k^2
+Q_{\dot\theta}\dot\theta_k^2
+R_FF_k^2
\right)
+J_N.
\]

注意与 inverse-dynamics 版本相比，这里没有：

\[
R_a\|a_k\|^2
\]

因为 `a_k` 根本不是独立优化变量。

## 8. 自适应时间步

与前一个工程保持一致：

\[
\Delta t_k=\Delta t_{min}\gamma^k,
\]

其中：

```text
N      = 20
DT_MIN = 0.020 s
DT_MAX = 0.045 s
```

这样近端预测时间分辨率更高，远端更稀疏。

## 9. 安装依赖

```bash
cd webots_forward_dynamics_mpc
python3 -m pip install -r requirements.txt
```

验证：

```bash
python3 -c "import casadi, numpy; print(casadi.__version__)"
```

如果 Webots 报 `No module named casadi`，让 Webots controller 使用安装了 CasADi 的同一个 Python 解释器。

## 10. 先运行离线验证

```bash
python3 tools/offline_validate.py
```

这里：

- MPC 内部采用 forward dynamics；
- MPC 每次只优化 `F_k` 和未来状态；
- 加速度由 `M^-1([F,0]-h)` 计算；
- 外部 plant 用独立 RK4 积分；
- Webots 中 RK4 plant 会被 Webots/ODE 替换。

成功时最后会看到：

```text
PASS: forward-dynamics MPC stabilized the nonlinear plant.
```

日志：

```text
logs/offline_validation.csv
```

## 11. Webots 运行方法

在 Webots 中打开：

```text
worlds/cartpole_forward_dynamics_mpc.wbt
```

然后点击 Run。

初始角度：

\[
\theta_0=0.08\;rad\approx4.58^\circ.
\]

控制目标：

\[
\theta\rightarrow0,
\qquad
x\rightarrow0.
\]

MPC 输出的第一个最优控制量直接发送：

```python
cart_motor.setForce(current_force)
```

Webots/ODE 是 MPC 外部的真实 plant。

## 12. 最重要的文件

先看：

```text
controllers/forward_dynamics_mpc/dynamics.py
```

核心：

```python
def forward_acceleration_ca(q, v, force):
    # M(q) = [[A, B], [B, D]], rhs = [F, 0] - h(q,v)
    det = A * D - B**2
    x_ddot = (D * rhs_1 - B * rhs_2) / det
    theta_ddot = (-B * rhs_1 + A * rhs_2) / det
    return ca.vertcat(x_ddot, theta_ddot)
```

然后看：

```text
controllers/forward_dynamics_mpc/mpc.py
```

核心只有：

```python
x_next = state_transition_ca(xk, uk, dt)
opti.subject_to(self.X[:, k + 1] == x_next)
```

这里和 inverse-dynamics 版本的最大区别是：

```text
没有 A 决策变量
没有 tau_id[1] == 0 这种 RNEA/path constraint
```

被动关节的 `0` 已经放进 forward dynamics 的输入向量：

\[
[F,0]^T,
\]

所以求出来的 `a` 天然服从欠驱动系统动力学。

## 13. Webots 参数必须和解析模型一致

如果修改 `.wbt` 里的：

- cart mass；
- pole mass；
- pole length；
- pole width；
- gravity；

必须同步修改：

```text
controllers/forward_dynamics_mpc/config.py
```

否则会引入 model mismatch。

## 14. 最适合和 inverse-dynamics 工程做的对比

保持两个工程的初始状态、预测时域和权重不变，然后比较：

1. 每次 NLP 的 decision variable 数量；
2. IPOPT 平均求解时间；
3. 收敛迭代次数；
4. 摆角恢复速度；
5. 小车最大位移；
6. 初始猜测较差时的鲁棒性；
7. 加大离散时间步以后两种 formulation 的差别。

在这个 2-DoF 问题上两者都很小，因此计算时间差不会像高维机器人那样明显；但它可以非常清楚地验证两种 formulation 的数学结构差异。
