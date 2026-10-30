import networkx as nx
import numpy as np
import random
from random import expovariate as EXP
import math
from tqdm import trange, tqdm
from main import simulate_SIS # 假设 simulate_SIS 函数在单独的文件中


def powerlaw_sample(alpha, tmin):
    xmin = tmin
    result = 999999999
    while result > 10000:
        u = random.random()
        result = xmin * math.pow(u, 1 / (1 - alpha))
    return result


# 网络参数和初始状态设置
N = 2500
k = 10
G = nx.random_regular_graph(k, N)

init_infected_rate = 0.3
pI0 = init_infected_rate
states = np.zeros(N)
for node in range(N):
    if random.random() < pI0:
        states[node] = 1

T = 50

# 三种分布的参数
distributions = {
    'Exponential': [{'beta': 0.26, 'mu': 0.5}],
    'Lognormal': [{'beta': 0.33, 'mu': 0, 'sigma': 1}],
    'Powerlaw': [{'beta': 0.3, 'lambda': 4, 't0': 1}]
}

num_repeats = 50


# 痊愈时间分布函数
def HT_DIST_exponential(mu):
    return np.random.exponential(1 / mu)


def HT_DIST_powerlaw(lambda_, t0):
    xmin = t0
    result = 999999999
    while result > 10000:
        u = np.random.uniform(0, 1)
        result = xmin * math.pow(u, 1 / (1 - lambda_))
    return result


def HT_DIST_lognormal(mu, sigma):
    return np.random.lognormal(mu, sigma)


# 存储所有分布的模拟结果
all_results = {}

for dist_name, params_list in distributions.items():
    all_results[dist_name] = []
    for params in params_list:
        group_results = []
        for _ in trange(num_repeats):
            states = np.zeros(N)
            for node in range(N):
                if random.random() < pI0:
                    states[node] = 1

            if dist_name == 'Exponential':
                mu = params['mu']
                HT_DIST = lambda: HT_DIST_exponential(mu)
            elif dist_name == 'Powerlaw':
                lambda_ = params['lambda']
                t0 = params['t0']
                HT_DIST = lambda: HT_DIST_powerlaw(lambda_, t0)
            elif dist_name == 'Lognormal':
                mu = params['mu']
                sigma = params['sigma']
                HT_DIST = lambda: HT_DIST_lognormal(mu, sigma)

            number_of_node_states, _ = simulate_SIS(T, params['beta'], HT_DIST, G, states.copy())

            times = number_of_node_states[0]
            num_I = np.array(number_of_node_states[2]) / N

            group_results.append((times, num_I))

        all_results[dist_name].append(group_results)

# 时间区间划分 - 均匀划分（用于标准差曲线）
time_interval_uniform = 0.1
time_points_uniform = np.arange(0, T + time_interval_uniform, time_interval_uniform)

# 时间区间划分 - 等比划分（用于95%置信区间带）
num_intervals_geometric = 500
time_points_geometric = []
q = 1.03
current_time = (q - 1) / (q ** num_intervals_geometric - 1) * 1724
while len(time_points_geometric) < num_intervals_geometric:
    time_points_geometric.append(current_time)
    current_time *= q
time_points_geometric = np.array(time_points_geometric[:num_intervals_geometric])

# 准备两种时间节点的数据
all_results_processed = {}

for dist_name, group_results_list in all_results.items():
    all_results_processed[dist_name] = []
    for group_results in group_results_list:
        processed_results = []
        for times, num_I in tqdm(group_results):
            times_uniform = np.array([min(time_points_uniform, key=lambda x: abs(x - t)) for t in times])
            times_geometric = np.array([min(time_points_geometric, key=lambda x: abs(x - t)) for t in times])
            processed_results.append((times_uniform, times_geometric, num_I))
        all_results_processed[dist_name].append(processed_results)

# 计算标准差曲线（基于均匀时间节点）
std_devs = {}
for dist_name in distributions:
    std_devs[dist_name] = []
for dist_name, group_results_list in tqdm(all_results_processed.items()):
    for group_results in group_results_list:
        time_dict_uniform = {t: [] for t in time_points_uniform}
        for times_uniform, _, num_I in group_results:
            for t, I in zip(times_uniform, num_I):
                time_dict_uniform[t].append(I)
        std_I = []
        for t in time_points_uniform:
            values = time_dict_uniform[t]
            if len(values) >= 2:
                std = np.std(values)
                std_I.append(std)
            else:
                std_I.append(np.nan)
        std_devs[dist_name].append(std_I)

# 计算95%置信区间带（基于等比时间节点）
means = {}
cis = {}
for dist_name in distributions:
    means[dist_name] = []
    cis[dist_name] = []
for dist_name, group_results_list in tqdm(all_results_processed.items()):
    for group_results in group_results_list:
        time_dict_geometric = {t: [] for t in time_points_geometric}
        for _, times_geometric, num_I in group_results:
            for t, I in zip(times_geometric, num_I):
                time_dict_geometric[t].append(I)
        mean_I = []
        ci_lower = []
        ci_upper = []
        for t in time_points_geometric:
            values = time_dict_geometric[t]
            if len(values) >= 2:
                mean = np.mean(values)
                values_sorted = np.sort(values)
                lower_bound = values_sorted[int(0.025 * len(values))]
                upper_bound = values_sorted[int(0.975 * len(values))]
                mean_I.append(mean)
                ci_lower.append(lower_bound)
                ci_upper.append(upper_bound)
            else:
                mean_I.append(np.nan)
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
        means[dist_name].append(mean_I)
        cis[dist_name].append((ci_lower, ci_upper))

# 保存结果到文件
np.savez(r'result/std_dev_and_ci/std_dev_and_ci.npz',
         time_points_uniform=time_points_uniform,
         time_points_geometric=time_points_geometric,
         std_devs=std_devs,
         means=means,
         cis=cis)
