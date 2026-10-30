import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 加载仿真结果
data = np.load(r'result/std_dev_and_ci/std_dev_and_ci.npz', allow_pickle=True)
time_points_uniform = data['time_points_uniform']
time_points_geometric = data['time_points_geometric']
std_devs = data['std_devs'].item()
means = data['means'].item()
cis = data['cis'].item()

distributions = ['Exponential', 'Lognormal', 'Powerlaw']
colors = {'Exponential': 'red', 'Lognormal': 'green', 'Powerlaw': 'blue'}
labels = {'Exponential': 'Exponential', 'Lognormal': 'Lognormal', 'Powerlaw': 'Powerlaw'}

# 绘制95%置信区间带
fig, axs = plt.subplots(3, 1, figsize=(12*2/2, 18/2))
for i, dist_name in enumerate(distributions):
    ax = axs[i]
    ax.grid(True)  # 这里确保每个子图都开启网格
    ax.tick_params(axis='both', labelsize=14 * 1.5)  # 统一修改刻度标签大小
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Times New Roman")  # 统一修改字体
    avg_mean = np.nanmean(means[dist_name], axis=0)
    avg_ci_lower = np.nanmean([ci[0] for ci in cis[dist_name]], axis=0)
    avg_ci_upper = np.nanmean([ci[1] for ci in cis[dist_name]], axis=0)

    # 处理NaN值
    valid_indices = ~np.isnan(avg_mean)
    valid_time_points = time_points_geometric[valid_indices]
    valid_avg_mean = avg_mean[valid_indices]
    valid_avg_ci_lower = avg_ci_lower[valid_indices]
    valid_avg_ci_upper = avg_ci_upper[valid_indices]
    ax.plot(valid_time_points, valid_avg_mean, label=f'{labels[dist_name]} Mean', color=colors[dist_name], linewidth=2)
    ax.fill_between(valid_time_points, valid_avg_ci_lower, valid_avg_ci_upper, alpha=0.3, color=colors[dist_name],
                    label=f'{labels[dist_name]} 95% CI')
    ax.set_xlabel('$t$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
    ax.set_ylabel('$\\rho_I$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
    ax.legend(prop={'size': 15}, loc='lower right')
    ax.set_xscale('log')

    # 添加嵌套标准差曲线小框
    inset_ax = ax.inset_axes([0.1, 0.3, 0.45, 0.45])  # 调整位置和大小
    avg_std = np.nanmean(std_devs[dist_name], axis=0)
    inset_ax.plot(time_points_uniform, avg_std, color=colors[dist_name], linestyle='-')
    inset_ax.set_title(f'{labels[dist_name]} Std Dev', fontsize=10)
    inset_ax.spines['top'].set_color('black')
    inset_ax.spines['bottom'].set_color('black')
    inset_ax.spines['left'].set_color('black')
    inset_ax.spines['right'].set_color('black')


plt.tight_layout()
plt.savefig(r"C:\Users\xy\Desktop\Python\SIS引擎\result\pictures\95.png", dpi=500, bbox_inches='tight')
plt.show()

