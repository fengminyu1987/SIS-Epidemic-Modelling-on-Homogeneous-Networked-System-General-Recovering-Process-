# -*- coding: utf-8 -*-
"""
Edge Switching Dynamics
Created on Wed Jul 12 18:09:33 2023
time(1->0): exp
time(0->1): pow
@author: zziya
"""
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import json
import copy

tmin = 50
tmax = 200
avgs = 10


def powerlaw_sample(alpha):
    xmin = 1
    result = 999999999
    while result > 10000:
        u = random.uniform(0, 1)
        result = xmin * math.pow(u, 1 / (1 - alpha))
    return result


def get_active_subgraph(G, edge_state):
    g_temp = copy.deepcopy(G)
    remove_set = [k for k, v in edge_state.items() if v == 1]
    g_temp.remove_edge_from(remove_set)
    return g_temp


N = 100
k = 4  # 4, 6, 10
beta = 0.6
Beta = np.arange(0, 0.30, 0.01)
gamma = 0.3
delta = 1
lam0 = 1
alpha1 = 2.6
network_type = 'rrg'
G = nx.random_regular_graph(k, N)
edge_set = list(G.edges)
result = {}
for beta in Beta:
    Fi_temp = []
    for avg in range(avgs):
        # initialization
        next_transition = {}
        edge_state = {}
        node_state = {}
        update_time = {}
        tnow = 0
        for i in edge_set:
            proba = random.uniform(0, 1)
            if proba < 0.5:
                edge_state[i] = 0
                next_transition[i] = powerlaw_sample(alpha1)
            else:
                edge_state[i] = 1
                next_transition[i] = random.expovariate(lam0)
        for i in G.nodes:
            node_state[i] = 0
            update_time[i] = random.expovariate(delta)
        node_state[random.choice(list(G.nodes))] = 1
        fi_temp = []

        while (tnow < tmax):
            state_transition_time = min(next_transition.values())
            update_strategy_time = min(update_time.values())
            if state_transition_time < update_strategy_time:
                changer = min(next_transition.items(), key=lambda x: x[1])[0]
                tnow = next_transition[changer]
                if edge_state[changer] == 0:  # become inactive
                    edge_state[changer] = 1
                    next_transition[changer] += random.expovariate(lam0)
                else:  # become active and update state
                    edge_state[changer] = 0
                    next_transition[changer] += powerlaw_sample(alpha1)
            else:
                updater = min(update_time.items(), key=lambda x: x[1])[0]
                tnow = update_time[updater]
                if node_state[updater] == 0:
                    Ni = 0
                    for node in G.neighbors(updater):
                        if (node, updater) in edge_set:
                            if edge_state[(node, updater)] == 0 and node_state[node] == 1:
                                Ni += 1
                        else:
                            if edge_state[(updater, node)] == 0 and node_state[node] == 1:
                                Ni += 1
                    proba = 1 - (1 - beta) ** Ni
                    if random.uniform(0, 1) < proba:
                        node_state[updater] = 1
                else:
                    if random.uniform(0, 1) < gamma:
                        node_state[updater] = 0
                update_time[updater] += random.expovariate(delta)
            fi = sum(node_state.values()) / N
            print('\r beta : {:.2f}, tnow : {:.2f}, fi : {:.3f}, avgs : {:d}'.format(beta, tnow, fi, avg), end='      ')
            if tnow > tmin:
                fi_temp.append(fi)
        Fi_temp.append(np.mean(fi_temp))
    Fi_result = np.mean(Fi_temp)
    # print(Fi_result)
    result['{:.2f}'.format(beta)] = Fi_result
json_str = json.dumps(result)
with open('.\\saves\\{:.2f}_{:.2f}_{}_{:d}.json'.format(lam0, alpha1, network_type, k), 'w') as json_file:
    json_file.write(json_str)
