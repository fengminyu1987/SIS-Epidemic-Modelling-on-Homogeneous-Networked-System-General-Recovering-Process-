import os
import math
import numpy as np
import networkx as nx
from random import random
from collections import Counter
from tqdm import tqdm
from main import simulate_SIS

# 定义用于传染行为等待时间的指数分布函数
def EXP(rate):
    # rate为传染率，对应平均等待时间为1/rate
    return np.random.exponential(1.0 / rate)

# 定义三种痊愈时间分布生成器
def HT_DIST_exponential(mu):
    return np.random.exponential(1.0 / mu)

def HT_DIST_powerlaw(lambda_, t0):
    xmin = t0
    result = 999999999
    while result > 10000:
        u = np.random.uniform(0, 1)
        result = xmin * math.pow(u, 1.0 / (1 - lambda_))
    return result

def HT_DIST_lognormal(mu, sigma):
    return np.random.lognormal(mu, sigma)


# 主实验设置
if __name__ == '__main__':
    T = 10
    N = 2500
    k = 10
    num_simulations = 50

    beta_exp = np.linspace(0.02, 0.5, 25)
    beta_logn = np.linspace(0.03, 0.6, 20)
    beta_pl = np.linspace(0.03, 0.66, 22)

    results = {
        'exp': {'beta': beta_exp, 'mean': [], 'lower': [], 'upper': []},
        'logn': {'beta': beta_logn, 'mean': [], 'lower': [], 'upper': []},
        'pl': {'beta': beta_pl, 'mean': [], 'lower': [], 'upper': []}
    }

    save_dir = os.path.join("result", "error_bar")
    os.makedirs(save_dir, exist_ok=True)

    for dist_key, beta_list, HT_DIST_func, dist_params in [
        ('exp', beta_exp, lambda: HT_DIST_exponential(0.5), {'label': 'Exponential'}),
        ('logn', beta_logn, lambda: HT_DIST_lognormal(0, 1), {'label': 'Lognormal'}),
        ('pl', beta_pl, lambda: HT_DIST_powerlaw(4, 1), {'label': 'Power-law'})
    ]:
        print(f"Running simulations for {dist_params['label']} distribution...")
        for beta in tqdm(beta_list):
            density_samples = []
            for _ in range(num_simulations):
                G = nx.random_regular_graph(k, N)
                states = np.zeros(N)
                for node in range(N):
                    if random() < 0.3:
                        states[node] = 1
                node_states, _ = simulate_SIS(T, beta, HT_DIST_func, G, states)
                # 如果记录的感染数据少于5个点，则直接认为最终感染数为0
                if len(node_states[2]) < 5:
                    final_I = 0
                else:
                    final_I = np.mean(node_states[2][-5:])
                density_samples.append(final_I / N)
            density_samples = np.array(density_samples)
            mean_density = np.mean(density_samples)
            lower_bound = np.percentile(density_samples, 16)
            upper_bound = np.percentile(density_samples, 84)
            results[dist_key]['mean'].append(mean_density)
            results[dist_key]['lower'].append(lower_bound)
            results[dist_key]['upper'].append(upper_bound)

    np.savez(os.path.join(save_dir, "simulation_results.npz"),
             exp_beta=results['exp']['beta'],
             exp_mean=np.array(results['exp']['mean']),
             exp_lower=np.array(results['exp']['lower']),
             exp_upper=np.array(results['exp']['upper']),
             logn_beta=results['logn']['beta'],
             logn_mean=np.array(results['logn']['mean']),
             logn_lower=np.array(results['logn']['lower']),
             logn_upper=np.array(results['logn']['upper']),
             pl_beta=results['pl']['beta'],
             pl_mean=np.array(results['pl']['mean']),
             pl_lower=np.array(results['pl']['lower']),
             pl_upper=np.array(results['pl']['upper'])
            )
    print("Simulation data saved to:", os.path.join(save_dir, "simulation_results.npz"))