import os
import math
import numpy as np
import networkx as nx
from random import random
from collections import Counter
from tqdm import trange


# 指数分布等待时间（传染过程）
def EXP(rate):
    return np.random.exponential(1.0 / rate)


# 三种痊愈时间分布生成器
def HT_DIST_exponential(mu):
    return np.random.exponential(1.0 / mu)


def HT_DIST_powerlaw(lambda_, t0):
    xmin = t0
    result = 1e9
    while result > 10000:
        u = np.random.uniform(0, 1)
        result = xmin * math.pow(u, 1.0 / (1 - lambda_))
    return result


def HT_DIST_lognormal(mu, sigma):
    return np.random.lognormal(mu, sigma)


# SIS模型仿真
def simulate_SIS(T, beta, HT_DIST, G, states):
    N = len(states)
    infect_times = np.zeros(N)
    next_events = np.zeros(N).tolist()
    number_of_node_states = [[], [], []]
    num_now = [0, 0]
    time = 0

    def become_I(node, time):
        states[node] = 1
        num_now[1] += 1
        num_now[0] -= 1
        infect_times[node] = T - time
        ht = HT_DIST()
        next_events[node] = (node, 'become S', time + ht)
        for neighbor in G.neighbors(node):
            it = EXP(beta)
            if it < ht and states[neighbor] == 0:
                next_events[neighbor] = (neighbor, 'become I', min(time + it, next_events[neighbor][2]))

    def become_S(node, time):
        states[node] = 0
        num_now[1] -= 1
        num_now[0] += 1
        infect_times[node] = 0
        IT = 999999999
        for neighbor in G.neighbors(node):
            it = EXP(beta)
            if states[neighbor] == 1 and time + it < next_events[neighbor][2]:
                IT = min(it, IT)
        next_events[node] = (node, 'become I', IT + time)

    num_now[1] = Counter(states)[1]
    num_now[0] = N - num_now[1]

    for node in range(N):
        if states[node] == 1:
            next_events[node] = (node, 'become I', time - infect_times[node])
        else:
            next_events[node] = (node, 'become S', time)
        states[node] = 1 - states[node]

    while time < T:
        event_now = min(next_events, key=lambda x: x[2])
        time = event_now[2]
        if event_now[1] == 'become S' and states[event_now[0]] == 1:
            become_S(event_now[0], event_now[2])
        if event_now[1] == 'become I' and states[event_now[0]] == 0:
            become_I(event_now[0], event_now[2])

    return number_of_node_states, infect_times.tolist()


# 仿真参数
T = 50
N = 50
num_simulations = 1
k_list = [4, 6, 8, 10, 12]  # 只保留偶数

distributions = {
    'exp': {'label': 'Exponential', 'HT_DIST': lambda: HT_DIST_exponential(0.5), 'E_W': 2},
    'logn': {'label': 'Lognormal', 'HT_DIST': lambda: HT_DIST_lognormal(0, 1), 'E_W': np.exp(0.5)},
    'pl': {'label': 'Power-law', 'HT_DIST': lambda: HT_DIST_powerlaw(4, 1), 'E_W': 1.5}
}

network_types = ['ER', 'BA', 'WS']

results = {net: {dist: {'k': [], 'tau': []} for dist in distributions.keys()} for net in network_types}

save_dir = os.path.join("result", "threshold")
os.makedirs(save_dir, exist_ok=True)

# 开始仿真
for net_type in network_types:
    for dist_key, dist_info in distributions.items():
        for k in k_list:
            print(f"Simulating {net_type} network, {dist_info['label']} recovery, k = {k}")

            # 生成不同的网络类型
            if net_type == 'ER':
                G = nx.random_regular_graph(k, N)
            elif net_type == 'BA':
                G = nx.barabasi_albert_graph(N, k // 2)  # BA网络 m = k/2
            elif net_type == 'WS':
                G = nx.watts_strogatz_graph(N, k, 0.1)  # WS网络重连概率0.1


            def init_states():
                states = np.zeros(N)
                for node in range(N):
                    if random() < 0.3:
                        states[node] = 1
                return states.copy()


            E_W = dist_info['E_W']
            beta = 1.0 / (k * E_W)
            max_iter = 100
            iter_count = 0
            sim_density = 0

            while iter_count < max_iter:
                density_samples = []
                for _ in trange(num_simulations):
                    states = init_states()
                    node_states, _ = simulate_SIS(T, beta, dist_info['HT_DIST'], G, states)
                    final_I = np.mean(node_states[2][-5:]) if len(node_states[2]) >= 5 else 0
                    density_samples.append(final_I / N)
                sim_density = np.mean(density_samples)
                if sim_density > 0.01:
                    break
                else:
                    beta += 0.01
                    iter_count += 1

            tau_sim = beta * E_W
            results[net_type][dist_key]['k'].append(k)
            results[net_type][dist_key]['tau'].append(tau_sim)
            print(
                f"  Network = {net_type}, k = {k:2d} : beta_threshold = {beta:.3f}, tau_sim = {tau_sim:.3f}, density = {sim_density:.4f}")

np.savez(os.path.join(save_dir, "simulation_threshold.npz"), **results)
print("Threshold simulation data saved to:", os.path.join(save_dir, "simulation_threshold.npz"))
