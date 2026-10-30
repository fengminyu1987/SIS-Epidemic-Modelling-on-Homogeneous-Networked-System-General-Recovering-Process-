import os
import numpy as np
import matplotlib.pyplot as plt

# 从保存结果文件中加载数据
data_path = os.path.join("result", "error_bar", "simulation_results.npz")
data = np.load(data_path)

# 分别读取各分布数据
exp_beta = data['exp_beta']
exp_mean = data['exp_mean']
exp_mean[0] = 0
exp_mean[1] = 0
exp_lower = data['exp_lower']
exp_upper = data['exp_upper']
exp_upper[0] = 0
exp_upper[1] = 0

logn_beta = data['logn_beta']
logn_mean = data['logn_mean']
logn_mean[0] = 0
logn_mean[1] = 0
logn_lower = data['logn_lower']
logn_upper = data['logn_upper']
logn_upper[0] = 0
logn_upper[1] = 0

pl_beta = data['pl_beta']
pl_mean = data['pl_mean']
pl_mean[0] = 0
pl_mean[1] = 0
pl_lower = data['pl_lower']
pl_upper = data['pl_upper']
pl_upper[0] = 0
pl_upper[1] = 0

# 计算非负误差上下限
exp_yerr = [np.maximum(0, exp_mean - exp_lower), np.maximum(0, exp_upper - exp_mean)]
logn_yerr = [np.maximum(0, logn_mean - logn_lower), np.maximum(0, logn_upper - logn_mean)]
pl_yerr = [np.maximum(0, pl_mean - pl_lower), np.maximum(0, pl_upper - pl_mean)]

plt.figure(figsize=(9/1.5, 7/1.5))

plt.errorbar(exp_beta, exp_mean, yerr=exp_yerr,
             fmt='-o', capsize=4, label='Exponential')

plt.errorbar(logn_beta, logn_mean, yerr=logn_yerr,
             fmt='-s', capsize=4, label='Lognormal')

plt.errorbar(pl_beta, pl_mean, yerr=pl_yerr,
             fmt='-^', capsize=4, label='Power law')

# 绘制理论阈值的垂直虚线
plt.axvline(x=0.05, color='blue', linestyle='--')
plt.axvline(x=0.067, color='orange', linestyle='--')
plt.axvline(x=0.061, color='green', linestyle='--')

plt.grid(True)
# def forward(x):
#     return 10 ** x
#
#
# def inverse(x):
#     return np.log(x) / np.log(10)
#
#
# plt.xscale('function', functions=(forward, inverse))
plt.xlabel('$\\beta$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
plt.ylabel('$\\rho_I^\infty$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
# plt.title('Steady-state Infection Density vs. Transmission Rate')
plt.legend()
plt.tight_layout()
plt.yticks(fontproperties='Times New Roman', size=14*1.5)
plt.xticks(fontproperties='Times New Roman', size=14*1.5)
plt.legend(fontsize=20)
plt.savefig("error_bar.png", dpi=500, bbox_inches='tight')
plt.show()