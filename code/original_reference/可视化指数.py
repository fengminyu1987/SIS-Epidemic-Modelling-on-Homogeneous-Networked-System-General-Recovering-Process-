from plot_distribution import plot_distribution as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import make_interp_spline

# ——————————导入数据———————————
k = 10
# 指数分布
mu = 0.2

# 幂律分布
alpha = 4
tmin = 5

# 均匀分布
a = 10
b = 50

#  0  1    2   3    4            5               6
# [N, k, beta, mu, avgs, init_infected_rate, simu_time]
dirlist_sit = os.listdir(r'结果/number_of_node_states/exponential/k=10/')
message_sit = [eval(i[0: -4]) for i in dirlist_sit]

dirlist_ET = os.listdir(r'result/Times/exponential/')
message_ET = [eval(i[0: -4]) for i in dirlist_ET]

print(message_ET)
ET_sample = [5, 9, 14]
# 1: 6,9   2:7,10,12,14

sit_sample = 12

sit = np.load(r'结果/number_of_node_states/exponential/k=10/' + dirlist_sit[sit_sample])
pIInfty = 1 - message_sit[sit_sample][3] / (message_sit[sit_sample][2] * message_sit[sit_sample][1])
print(pIInfty)
N = message_sit[sit_sample][0]

Ts = [np.load(r'result/Times/exponential/' + dirlist_ET[i]) for i in ET_sample]
print([message_ET[i] for i in ET_sample])
print(message_sit[sit_sample])
print(dirlist_ET)
# ————————————————————————————


# ——————————————————————————————————————可视化结果—————————————————————————————————————————————
t1 = [i for i in Ts[0] if i > 0]
t2 = [i for i in Ts[1] if i > 0]
t3 = [i for i in Ts[2] if i > 0]
labels = ['$\mu='+str(message_ET[i][3])+'$' for i in ET_sample]
pd([t1, t2, t3], 50, labels)
from KL散度 import convergent_power_law_pdf as cpl
from scipy.stats import expon
from KL散度 import ladder_pdf

m = min(max(t1), max(t2), max(t3))
t = np.arange(0, m, m / 1000)

# cpl_Theoretical = cpl(t, alpha, tmin)
# EXP_Theoretical = expon.pdf(t, scale=1 / mu)
# ladder_Theoretical = ladder_pdf(t, a, b)

# plt.plot(t, cpl_Theoretical, color='y', linestyle='--', label='Theoretical')
# plt.plot(t, EXP_Theoretical, color='y', linestyle='--')
# plt.plot(t, ladder_Theoretical, color='y', linestyle='--')
Theoreticals = [expon.pdf(t, scale=1 / message_ET[i][3]) for i in ET_sample]
plt.plot(t, Theoreticals[0], color='r', linestyle='--')
plt.plot(t, Theoreticals[1], color='g', linestyle='--')
plt.plot(t, Theoreticals[2], color='b', linestyle='--')

plt.grid(True)
# plt.legend(fontsize=15)
plt.legend(prop={'size': 30})

plt.xlim(0, m)
plt.savefig(r"C:\Users\xy\Desktop\Python\SIS引擎\result\pictures\exp\EXP_Tinfty_scatter.png", dpi=500, bbox_inches='tight')
print('EXP_Tinfty_scatter.png保存成功！')
plt.show()

plt.plot(sit[0], sit[2] / N, color='r', linewidth=2, label='$\\rho_I(t)$')
plt.axhline(pIInfty, color=(1, 0, 0, 0.5), linestyle='--', linewidth=2)

plt.plot(sit[0], sit[1] / N, color='g', linewidth=2, label='$\\rho_S(t)$')
plt.axhline(1 - pIInfty, color=(0, 1, 0, 0.5), linestyle='--', linewidth=2)

from 幂率数值解 import plot_grp_sis_model

plot_grp_sis_model(dist="exponential", dt=0.01, T_max=50, mu_exp=0.5, beta=0.26)

# plt.title('$\mathbf{Images \: of \: changes \: in \: infection/susceptibility \: density}$')
plt.xlabel('$t$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
plt.ylabel('$\mathbf{Density}$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
plt.yticks(fontproperties='Times New Roman', size=14*1.5)
plt.xticks(fontproperties='Times New Roman', size=14*1.5)
plt.xscale('log')
plt.grid(True)
# plt.legend()
plt.legend(prop={'size': 20})
plt.savefig(r"C:\Users\xy\Desktop\Python\SIS引擎\result\pictures\exp\EXP_2500_10_0.26_0.5_50_0.3_50.png", dpi=500, bbox_inches='tight')
print('EXP_2500_10_0.26_0.5_50_0.3_50.png保存成功！')
plt.show()

from KL散度 import check_exponential_distribution, check_convergent_power_law_distribution, check_ladder_distribution

# 计算KL散度
kl_div = check_exponential_distribution(t1, message_ET[ET_sample[0]][3])
print("\nt1 KL散度：", kl_div)
kl_div = check_exponential_distribution(t2, message_ET[ET_sample[1]][3])
print("\nt2 KL散度：", kl_div)
kl_div = check_exponential_distribution(t3, message_ET[ET_sample[2]][3])
print("\nt3 KL散度：", kl_div)


def smooth(xy):
    m = make_interp_spline(xy[0], xy[1])
    xs = np.linspace(min(xy[0]), max(xy[0]), 500)
    ys = m(xs)
    return [xs, ys]


xy = np.load(r'结果/number_of_infection_vs_beta/exponential/k=4_mu=0_5.npy')
xy = [xy[0], np.array([xy[1][i] if xy[0][i] > 0.5/4 else 0 for i in range(len(xy[1]))])]
xy1 = np.load(r'结果/number_of_infection_vs_beta/exponential/k=6_mu=0_5.npy')
xy1 = [xy1[0], np.array([xy1[1][i] if xy1[0][i] > 0.5/6 else 0 for i in range(len(xy1[1]))])]
xy2 = np.load(r'结果/number_of_infection_vs_beta/exponential/k=10_mu=0_5.npy')
xy2 = [xy2[0], np.array([xy2[1][i] if xy2[0][i] > 0.5/10 else 0 for i in range(len(xy2[1]))])]

plt.plot(xy[0], xy[1] / message_sit[-1][0], color='r', label='$\langle k\\rangle=4$', marker='.')
plt.axvline(0.5/4, color='r', linestyle='--', linewidth=2, alpha=0.5)
plt.plot(xy1[0], xy1[1] / message_sit[-1][0], color='g', label='$\langle k\\rangle=6$', marker='.')
plt.axvline(0.5/6, color='g', linestyle='--', linewidth=2, alpha=0.5)
plt.plot(xy2[0], xy2[1] / message_sit[-1][0], color='b', label='$\langle k\\rangle=10$', marker='.')
plt.axvline(0.5/10, color='b', linestyle='--', linewidth=2, alpha=0.5)


def pIInfty_of_EXP(betas, mu, AVG_k):
    result = np.zeros(len(betas))
    for i in range(len(betas)):
        result[i] = max(1 - mu / (betas[i] * AVG_k), 0)
    return result


def pIInfty_of_power_law(betas, alpha, tmin, AVG_k):
    result = np.zeros(len(betas))
    for i in range(len(betas)):
        result[i] = max(1 - (alpha - 2) / (betas[i] * AVG_k * (alpha - 1) * tmin), 0)
    return result


def pIInfty_of_uniform(betas, a, b, AVG_k):
    result = np.zeros(len(betas))
    for i in range(len(betas)):
        result[i] = max(1 - 2 / (betas[i] * AVG_k * (a + b)), 0)
    return result


x = np.arange(min(xy[0]), max(xy[0]), 1/500)

# plt.plot(x, pIInfty_of_EXP(x, 0.5, 4), color='orange', label='Theoretical', linestyle='--')
# plt.plot(x, pIInfty_of_EXP(x, 0.5, 6), color='orange', linestyle='--')
# plt.plot(x, pIInfty_of_EXP(x, 0.5, 10), color='orange', linestyle='--')
plt.grid(True)
plt.xlabel('$\\beta$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})
def forward(x):
    return 10 ** x


def inverse(x):
    return np.log(x) / np.log(10)


plt.yticks(fontproperties='Times New Roman', size=14*1.5)
plt.xticks(fontproperties='Times New Roman', size=14*1.5)
plt.xscale('function', functions=(forward, inverse))
plt.ylabel('$\\rho_I^\infty$', fontdict={'family': 'Times New Roman', 'size': 16*1.5})

# plt.legend()
plt.legend(prop={'size': 20})
plt.savefig(r"C:\Users\xy\Desktop\Python\SIS引擎\result\pictures\exp\EXP_pIInfty.png", dpi=500, bbox_inches='tight')
print('EXP_pIInfty.png保存成功！')
plt.show()

# ————————————————————————————————————————————————————————————————————————————————————————————
