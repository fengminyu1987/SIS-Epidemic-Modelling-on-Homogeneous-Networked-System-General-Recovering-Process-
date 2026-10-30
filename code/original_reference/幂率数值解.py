import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from tqdm import trange


def plot_grp_sis_model(dist="lognormal", beta=0.33, k_avg=10, T_max=50.0, dt=0.1, **kwargs):
    """
    使用 grp-SIS 模型模拟感染密度随时间的变化并绘制图像。

    参数:
      dist (str): 指定恢复分布类型，可取 "lognormal", "powerlaw", "exponential"。
      beta (float): 感染率，默认值为 0.33。
      k_avg (int): 网络的平均度，默认值为 10。
      T_max (float): 模拟终止时刻，默认值为 50.0。
      dt (float): 时间步长，默认值为 0.1。
      kwargs: 其他分布参数：
          当 dist=="lognormal" 时，需提供 mu_lognorm (均值，默认 0.0) 和 sigma_lognorm (标准差，默认 1.0)；
          当 dist=="powerlaw" 时，需提供 lambda_pl (幂律指数，默认 4) 和 t0_pl (下界，默认 1)；
          当 dist=="exponential" 时，需提供 mu_exp (指数分布参数，默认 0.5)。
    """
    # 计算总时间步数
    N = int(T_max / dt) + 1
    tau_grid = np.linspace(0, T_max, N)

    # 根据不同分布定义 w(τ) 和 F0(τ)
    if dist == "lognormal":
        mu_lognorm = kwargs.get("mu_lognorm", 0.0)
        sigma_lognorm = kwargs.get("sigma_lognorm", 1.0)

        def w_tau(tau):
            if tau <= 0:
                return 0.0
            else:
                return 1 / (tau * np.sqrt(2 * np.pi) * sigma_lognorm) * np.exp(
                    -((np.log(tau) - mu_lognorm) ** 2) / (2 * sigma_lognorm ** 2))

        # F0(τ)=P(W>τ)
        F0_vals = np.array(
            [0.5 * erfc((np.log(tau) - mu_lognorm) / (np.sqrt(2) * sigma_lognorm)) if tau > 0 else 1.0 for tau in
             tau_grid])

    elif dist == "powerlaw":
        lambda_pl = kwargs.get("lambda_pl", 4)
        t0_pl = kwargs.get("t0_pl", 1)

        def w_tau(tau):
            # 当 tau 小于下界 t0_pl 时，概率密度为 0
            if tau <= t0_pl:
                return 0.0
            else:
                return (lambda_pl - 1) / t0_pl * (tau / t0_pl) ** (-lambda_pl)

        # 注意：对于 tau <= t0_pl，通常 F0=1；tau > t0_pl 则 F0 = (tau/t0_pl)^(1-lambda_pl)
        F0_vals = np.array([1.0 if tau <= t0_pl else (tau / t0_pl) ** (1 - lambda_pl) for tau in tau_grid])

    elif dist == "exponential":
        mu_exp = kwargs.get("mu_exp", 0.5)

        def w_tau(tau):
            if tau < 0:
                return 0.0
            else:
                return mu_exp * np.exp(-mu_exp * tau)

        F0_vals = np.array([np.exp(-mu_exp * tau) for tau in tau_grid])

    else:
        raise ValueError("未知的分布类型，请选择 'lognormal', 'powerlaw' 或 'exponential'.")

    # 对所有 τ 预计算 w(τ)
    w_vals = np.array([w_tau(tau) for tau in tau_grid])

    # 初始化 ρ_I 数组（行：时间步，列：τ 的离散点）。
    # 注意：我们只使用三角区域（对于 t_n，仅 j=0,...,n 有意义）
    rhoI = np.zeros((N, N))
    # 初始条件：t=0 时，所有 τ 处均设为 0.3（新感染患者）
    rhoI[0, :] = 0.3

    # 主循环：沿时间步更新
    for n in trange(0, N - 1):
        # 当前时间 t_n 整体感染密度 I(t_n) 取 rhoI[n, n]，易感密度 S(t_n)=1 - I(t_n)
        I_n = rhoI[n, n]
        S_n = 1 - I_n
        source = beta * k_avg * S_n * I_n  # 新感染率

        # 边界条件：新感染在 τ=0 处注入
        rhoI[n + 1, 0] = source * dt

        # 利用累加和计算积分项 Q
        cum_sum = 0.0
        for j in range(0, n + 1):
            if j > 0:
                cum_sum += (w_vals[j - 1] / F0_vals[j - 1]) * (rhoI[n, j] - rhoI[n, j - 1])
            # 沿特征线更新：rhoI[n+1, j+1] = rhoI[n, j] + dt*(source - Q)
            rhoI[n + 1, j + 1] = rhoI[n, j] + dt * (source - cum_sum)

    # 整体感染密度 I(t) 取 ρ_I(t;t)，即对角线的值
    t_vals = np.linspace(0, T_max, N)
    I_vals = np.array([rhoI[n, n] for n in range(N)])

    # 绘图
    # plt.figure(figsize=(10, 6))
    plt.plot(t_vals, I_vals, label="$\\rho_I^n(t)$", color='blue', linewidth=5, alpha=0.3)
    plt.plot(t_vals, 1 - I_vals, label="$\\rho_S^n(t)$", color='purple', linewidth=5, alpha=0.3)
    # plt.xlabel("时间 t")
    # plt.ylabel("感染密度")
    # plt.title("grp-SIS 模型下的感染密度随时间变化")
    # plt.legend()
    # plt.grid(True)
    # plt.xscale('log')
    # plt.show()


if __name__ == '__main__':
    # 示例调用：分别绘制三种恢复分布下的结果
    # 对数正态分布
    plot_grp_sis_model(dist="lognormal", dt=0.01, T_max=5, mu_lognorm=0.0, sigma_lognorm=1.0)

    # 幂律分布
    plot_grp_sis_model(dist="powerlaw", dt=0.01, T_max=5, lambda_pl=4, t0_pl=1)

    # 指数分布
    plot_grp_sis_model(dist="exponential", dt=0.01, T_max=5, mu_exp=0.5)
