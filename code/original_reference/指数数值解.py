import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm import trange

# 参数设置
beta = 0.26
k_avg = 10  # 平均度（可根据需要调整）
T_max = 50.0  # 模拟终止时刻
dt = 0.05  # 时间步长
N = int(T_max / dt) + 1  # 总时间步数

# 构造 τ 网格（0到T_max，步长 dt）
tau_grid = np.linspace(0, T_max, N)  # 共 N 个点

# 指数分布参数（mu = 0.5）
mu_exp = 0.5


def w_tau(tau):
    if tau <= 0:
        return 0.0
    else:
        return mu_exp * np.exp(-mu_exp*tau)


# 预计算 w(τ) 和 F0(τ)（F0(τ)=P(W>τ)）
w_vals = np.array([w_tau(tau) for tau in tau_grid])
F0_vals = np.array(
    [np.exp(-mu_exp*tau) if tau > 0 else 1.0 for tau in tau_grid])

# 初始化 ρ_I 数组，行：时间步，列：τ 的离散点（仅使用三角区域：对于 t_n，有 j=0,...,n）
rhoI = np.zeros((N, N))
# 初始条件：t=0 时，对所有 τ 都设为 0.3
rhoI[0, :] = 0.3

# 主循环：沿时间步更新
for n in trange(0, N - 1):
    # 对于时刻 t_n，有效的 τ 范围为 j=0,...,n
    # 令当前整体感染密度 I(t_n)=ρ_I^{n,n}，故易感密度 S(t_n)=1 - ρ_I^{n,n}
    I_n = rhoI[n, n]
    S_n = 1 - I_n
    source = beta * k_avg * S_n * I_n  # 新感染率（单位：密度/时间）

    # 边界条件：新感染在 τ=0 处注入
    rhoI[n + 1, 0] = source * dt

    # 对于内部点：利用沿特征线的更新
    for j in range(0, n + 1):
        # 计算积分项 Q(t_n,τ_j) 用离散累加法（前向差分近似 dρ/dτ）
        Q = 0.0
        for m in range(0, j):
            Q += (w_vals[m] / F0_vals[m]) * (rhoI[n, m + 1] - rhoI[n, m])
        # 更新公式：ρ_I^{n+1, j+1} = ρ_I^{n,j} + dt*(source - Q)
        rhoI[n + 1, j + 1] = rhoI[n, j] + dt * (source - Q)

# 整体感染密度函数为 I(t)=ρ_I(t;t)，即取对角线
t_vals = np.linspace(0, T_max, N)
I_vals = np.array([rhoI[n, n] for n in range(N)])

# 绘图展示感染密度随时间的变化
plt.figure(figsize=(10, 6))
plt.plot(t_vals, I_vals, label="感染密度 ρ_I(t,t)")
plt.xlabel("时间 t")
plt.ylabel("感染密度")
plt.title("grp-SIS 模型下的感染密度随时间变化")
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.show()
