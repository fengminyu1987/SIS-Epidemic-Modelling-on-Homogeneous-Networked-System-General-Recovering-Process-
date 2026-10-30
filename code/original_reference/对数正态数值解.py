# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import erfc
# from tqdm import trange
#
# # 参数设置
# beta = 0.33
# k_avg = 10  # 平均度（可根据需要调整）
# T_max = 50.0  # 模拟终止时刻
# dt = 0.1  # 时间步长
# N = int(T_max / dt) + 1  # 总时间步数
#
# # 构造 τ 网格（0到T_max，步长 dt）
# tau_grid = np.linspace(0, T_max, N)  # 共 N 个点
#
# # 对数正态分布参数（μ=0, σ=1）
# mu_lognorm = 0.0
# sigma_lognorm = 1.0
#
#
# def w_tau(tau):
#     if tau <= 0:
#         return 0.0
#     else:
#         return 1 / (tau * np.sqrt(2 * np.pi) * sigma_lognorm) * np.exp(
#             -((np.log(tau) - mu_lognorm) ** 2) / (2 * sigma_lognorm ** 2))
#
#
# # 预计算 w(τ) 和 F0(τ)（F0(τ)=P(W>τ)）
# w_vals = np.array([w_tau(tau) for tau in tau_grid])
# F0_vals = np.array(
#     [0.5 * erfc((np.log(tau) - mu_lognorm) / (np.sqrt(2) * sigma_lognorm)) if tau > 0 else 1.0 for tau in tau_grid])
#
# # 初始化 ρ_I 数组，行：时间步，列：τ 的离散点（仅使用三角区域：对于 t_n，有 j=0,...,n）
# rhoI = np.zeros((N, N))
# # 初始条件：t=0 时，对所有 τ 都设为 0.3
# rhoI[0, :] = 0.3
#
# # 主循环：沿时间步更新
# for n in trange(0, N - 1):
#     # 对于时刻 t_n，有效的 τ 范围为 j=0,...,n
#     # 令当前整体感染密度 I(t_n)=ρ_I^{n,n}，故易感密度 S(t_n)=1 - ρ_I^{n,n}
#     I_n = rhoI[n, n]
#     S_n = 1 - I_n
#     source = beta * k_avg * S_n * I_n  # 新感染率（单位：密度/时间）
#
#     # 边界条件：新感染在 τ=0 处注入
#     rhoI[n + 1, 0] = source * dt
#
#     # 对于内部点：利用沿特征线的更新
#     for j in range(0, n + 1):
#         # 计算积分项 Q(t_n,τ_j) 用离散累加法（前向差分近似 dρ/dτ）
#         Q = 0.0
#         for m in range(0, j):
#             Q += (w_vals[m] / F0_vals[m]) * (rhoI[n, m + 1] - rhoI[n, m])
#         # 更新公式：ρ_I^{n+1, j+1} = ρ_I^{n,j} + dt*(source - Q)
#         rhoI[n + 1, j + 1] = rhoI[n, j] + dt * (source - Q)
#
# # 整体感染密度函数为 I(t)=ρ_I(t;t)，即取对角线
# t_vals = np.linspace(0, T_max, N)
# I_vals = np.array([rhoI[n, n] for n in range(N)])
#
# # 绘图展示感染密度随时间的变化
# plt.figure(figsize=(10, 6))
# plt.plot(t_vals, I_vals, label="感染密度 ρ_I(t,t)")
# plt.xlabel("时间 t")
# plt.ylabel("感染密度")
# plt.title("grp-SIS 模型下的感染密度随时间变化")
# plt.legend()
# plt.grid(True)
# plt.xscale('log')
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm import trange

def plot_grp_sis_model(beta=0.33, k_avg=10, T_max=50.0, dt=0.1, mu_lognorm=0.0, sigma_lognorm=1.0):
    """
    使用 grp-SIS 模型模拟感染密度随时间的变化并绘制图像。

    参数:
    beta (float): 感染率，默认值为 0.33
    k_avg (int): 网络的平均度，默认值为 10
    T_max (float): 模拟终止时刻，默认值为 50.0
    dt (float): 时间步长，默认值为 0.1
    mu_lognorm (float): 对数正态分布的均值，默认值为 0.0
    sigma_lognorm (float): 对数正态分布的标准差，默认值为 1.0
    """
    # 计算总时间步数
    N = int(T_max / dt) + 1

    # 构造 τ 网格（0 到 T_max，步长 dt）
    tau_grid = np.linspace(0, T_max, N)

    # 定义恢复时间分布的概率密度函数 w(τ)
    def w_tau(tau):
        if tau <= 0:
            return 0.0
        else:
            return 1 / (tau * np.sqrt(2 * np.pi) * sigma_lognorm) * np.exp(-((np.log(tau) - mu_lognorm) ** 2) / (2 * sigma_lognorm ** 2))

    # 预计算 w(τ) 和 F0(τ)（F0(τ)=P(W>τ)）
    w_vals = np.array([w_tau(tau) for tau in tau_grid])
    F0_vals = np.array([0.5 * erfc((np.log(tau) - mu_lognorm) / (np.sqrt(2) * sigma_lognorm)) if tau > 0 else 1.0 for tau in tau_grid])

    # 初始化 ρ_I 数组，行：时间步，列：τ 的离散点
    rhoI = np.zeros((N, N))
    # 初始条件：t=0 时，对所有 τ 都设为 0.3
    rhoI[0, :] = 0.3

    # 主循环：沿时间步更新
    for n in trange(0, N - 1):
        # 对于时刻 t_n，有效的 τ 范围为 j=0,...,n
        # 当前整体感染密度 I(t_n)=rhoI[n, n]，故易感密度 S(t_n)=1 - rhoI[n, n]
        I_n = rhoI[n, n]
        S_n = 1 - I_n
        source = beta * k_avg * S_n * I_n  # 新感染率（单位：密度/时间）

        # 边界条件：新感染在 τ=0 处注入
        rhoI[n + 1, 0] = source * dt

        # 利用累加和更新 Q，减少重复求和
        cum_sum = 0.0
        # 对于内部点：j 从 0 到 n
        for j in range(0, n + 1):
            if j > 0:
                # 只增量更新累加和，避免重复求和
                cum_sum += (w_vals[j - 1] / F0_vals[j - 1]) * (rhoI[n, j] - rhoI[n, j - 1])
            # 更新公式：rhoI[n+1, j+1] = rhoI[n, j] + dt*(source - cum_sum)
            rhoI[n + 1, j + 1] = rhoI[n, j] + dt * (source - cum_sum)

    # 整体感染密度函数为 I(t)=rhoI(t;t)，即取对角线
    t_vals = np.linspace(0, T_max, N)
    I_vals = np.array([rhoI[n, n] for n in range(N)])

    # 绘图展示感染密度随时间的变化
    # plt.figure(figsize=(10, 6))
    plt.plot(t_vals, I_vals, label="$\\rho_I^n(t)$", color='blue', linewidth=5, alpha=0.3)
    plt.plot(t_vals, 1-I_vals, label='$\\rho_S^n(t)$', color='purple', linewidth=5, alpha=0.3)


if __name__ == '__main__':
    # 调用函数绘制图像
    plot_grp_sis_model(dt=0.01, T_max=5)
    plt.xlabel("t")
    plt.ylabel("rho")
    plt.title("grp-SIS")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.show()
