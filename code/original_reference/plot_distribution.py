import matplotlib.pyplot as plt
import numpy as np
from random import expovariate as EXP


# def plot_distribution(data, n):
#     # 计算区间的边界
#     min_val = min(data)
#     max_val = max(data)
#     interval_size = (max_val - min_val) / n
#     bins = [min_val + i * interval_size for i in range(n)]
#     bins.append(max_val)  # 添加最后一个区间的边界
#
#     # 使用 numpy 的 histogram 函数计算频数
#     hist, _ = np.histogram(data, bins=bins)
#
#     # 计算每个区间的中点和频率
#     bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n)]
#     frequencies = hist / len(data)  # 计算频率，即频数除以总数
#
#     # 绘制折线图
#     plt.figure(figsize=(8, 6))
#     plt.plot(bin_centers, frequencies, marker='o', linestyle='-', color='r', linewidth=3)
#     plt.title('Distribution Plot')
#     plt.xlabel('Value')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.show()

def plot_distribution(Datas, n, labels):
    colors = ['r', 'g', 'b']*100
    dots = ['o', 'x', '*']*100
    # labels = ['sigma=0.5', 'sigma=0.75', 'sigma=1']
    plt.figure(figsize=(11 * 4/3, 10.5 * 2/3))
    for i in range(len(Datas)):
        # 计算区间的边界
        min_val = min(Datas[i])
        max_val = max(Datas[i])
        # n = max_val - min_val
        interval_size = (max_val - min_val) / n
        bins = [min_val + i * interval_size for i in range(n)]
        bins.append(max_val)  # 添加最后一个区间的边界

        # 使用 numpy 的 histogram 函数计算频数
        hist, _ = np.histogram(Datas[i], bins=bins)

        # 计算每个区间的中点和频率
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(n)]
        frequencies = hist / (len(Datas[i]) * interval_size)  # 计算频率，即频数除以总数

        # 绘制折线图

        plt.plot(bin_centers, frequencies, marker=dots[i], linestyle='', color=colors[i], linewidth=3, label=labels[i], markersize=15)

        # # 绘制平均值的垂直线
        # mean_value = np.mean(Datas[i])
        # plt.axvline(mean_value, color=colors[i], linestyle='--', linewidth=2, alpha=0.5)

    # plt.title('$\mathbf{Distribution \: of \: }T(\infty)$')
    plt.xlabel('$\\tau$', fontdict={'family': 'Times New Roman', 'size': 26*1.5})
    plt.ylabel('$\mathbf{PDF}$', fontdict={'family': 'Times New Roman', 'size': 26*1.5})
    plt.yticks(fontproperties='Times New Roman', size=30)
    plt.xticks(fontproperties='Times New Roman', size=30)

    # # 将坐标轴改为对数坐标
    # plt.xscale('log')
    # plt.yscale('log')

    plt.grid(True)
    # plt.legend(prop={'size': 40})
    # plt.show()
    return frequencies, min_val, max_val


if __name__ == '__main__':
    # 示例用法
    data = []
    for _ in range(10000):
        data.append(EXP(2))
    plot_distribution(data, 50)
