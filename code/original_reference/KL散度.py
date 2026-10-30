import numpy as np
from scipy.stats import expon, entropy, lognorm
from math import erfc, log, sqrt, exp


def logerfc_pdf(bin_centers, mu, sigma):
    result = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        if bin_centers[i] > 0:
            result[i] = erfc((log(bin_centers[i])-mu) / (sqrt(2) * sigma)) / (2 * exp(mu + sigma ** 2 / 2))
        elif bin_centers[i] == 0:
            result[i] = exp(-mu - sigma ** 2 / 2)
    return result


def convergent_power_law_pdf(bin_centers, alpha, tmin):
    result = np.zeros(len(bin_centers))
    for i in range(len(bin_centers)):
        result[i] = (alpha - 2) / ((alpha - 1) * tmin)
        if bin_centers[i] > tmin:
            result[i] *= (bin_centers[i]/tmin) ** (1-alpha)
        if bin_centers[i] < 0:
            result[i] *= 0
    return result


def ladder_pdf(t, a, b):
    result = np.zeros(len(t))
    for i in range(len(t)):
        result[i] = 2 / (a + b)
        if a < t[i] <= b:
            result[i] *= (b - t[i])/(b - a)
        if t[i] < 0:
            result[i] *= 0
    return result


def kl_divergence(p, q):
    """计算两个分布的KL散度"""
    return entropy(p, q)


def check_exponential_distribution(ttt, mu, num_bins=50):
    # 计算样本ttt的直方图分布
    hist, bin_edges = np.histogram(ttt, bins=num_bins, density=True)

    # 计算bin的中心点
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 基于给定的mu，计算指数分布的理论概率密度
    q = expon.pdf(bin_centers, scale=1 / mu)

    # 计算KL散度
    kl_div = kl_divergence(hist, q)

    return kl_div


def check_convergent_power_law_distribution(ttt, alpha, tmin, num_bins=50):
    # 计算样本ttt的直方图分布
    hist, bin_edges = np.histogram(ttt, bins=num_bins, density=True)

    # 计算bin的中心点
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 基于给定的alpha/tmin，计算理论概率密度
    q = convergent_power_law_pdf(bin_centers, alpha, tmin)

    # 计算KL散度
    kl_div = kl_divergence(hist, q)

    return kl_div


def check_ladder_distribution(ttt, a, b, num_bins=50):
    # 计算样本ttt的直方图分布
    hist, bin_edges = np.histogram(ttt, bins=num_bins, density=True)

    # 计算bin的中心点
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 基于给定的alpha/tmin，计算理论概率密度
    q = ladder_pdf(bin_centers, a, b)

    # 计算KL散度
    kl_div = kl_divergence(hist, q)

    return kl_div


def check_ladder_distribution(ttt, a, b, num_bins=50):
    # 计算样本ttt的直方图分布
    hist, bin_edges = np.histogram(ttt, bins=num_bins, density=True)

    # 计算bin的中心点
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    q = ladder_pdf(bin_centers, a, b)

    # 计算KL散度
    kl_div = kl_divergence(hist, q)

    return kl_div


def check_logerfc_distribution(ttt, mu, sigma, num_bins=50):
    # 计算样本ttt的直方图分布
    hist, bin_edges = np.histogram(ttt, bins=num_bins, density=True)

    # 计算bin的中心点
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    q = logerfc_pdf(bin_centers, mu, sigma)

    # 计算KL散度
    kl_div = kl_divergence(hist, q)

    return kl_div

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.arange(-1, 10, 0.01)
    plt.plot(x, logerfc_pdf(x, 0, 0.2))
    plt.show()
