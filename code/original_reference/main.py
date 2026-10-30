import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import expovariate as EXP
from random import random
import math
from plot_distribution import plot_distribution as pd
from collections import Counter


def powerlaw_sample(alpha, tmin):
    xmin = tmin
    result = 999999999
    while result > 10000:
        u = random()
        result = xmin * math.pow(u, 1 / (1 - alpha))
    return result


def simulate_SIS(T, beta, HT_DIST, G, states, infect_times=np.zeros(2500)):
    """
    一般痊愈过程SIS模型的模拟，其中感染节点对其邻居的传染行为仍是泊松过程。因为需记录感染时间，所以不是epsilon-SIS模型，因为随机加入的感染节点会
    扰乱时间分布。（虽说在epsilon极小时可近似）
    :param T: 模拟运行时间。
    :param beta: 描述传染行为的泊松过程的强度，也是对单体的传染速率，其倒数就是单一易感节点被单一感染节点传染的平均时间。
    :param HT_DIST:heal time distribution，即痊愈时间分布，是节点从被感染开始到完全痊愈所消耗时间的分布。应与系统状态量独立。是一个函数，返回满足某个分布的随机数。
    :param G:一个同质网络。
    :param states:节点的初始状态列表，0为易感态，1为感染态。
    :param infect_times:节点的初始已感染时间。对于易感节点，其值为0，统计时需将0值排除。
    :return: number_of_node_states:模拟中所有事件发生时节点的状态数量记录。用于绘制感染密度变化图像。
             infect_times:运行结束时节点的已感染时间。
    """
    # ——————————————————————————————————————参数————————————————————————————————————————————
    N = len(states)  # 节点数量
    next_events = np.zeros(N).tolist()  # 储存节点下一次要发生的事件信息
    infect_times = np.zeros(N)  # 假设感染节点都是刚刚被感染的状态
    number_of_node_states = [[], [], []]  # 记录状态数量及其对应的时间，元素是(time, (numS, numI))
    num_now = [0, 0]  # 当前状态数量
    time = 0  # 当前时间

    # ——————————————————————————————————————————————————————————————————————————————————————

    def become_I(node, time):
        """
        节点的感染函数，实现节点被感染的效果。
        :param node: 节点的序号。
        :param time: 节点的被感染时间。
        :return: 无
        """
        states[node] = 1  # 将该节点状态变为感染态
        if time != 0:
            num_now[1] += 1  # 当前感染节点数加一
            num_now[0] -= 1

        infect_times[node] = T - time  # 因为求的是模拟结束时的已感染时间，于是其一定等于T减去上一次被感染的时
        # 间。若该节点痊愈，则改为0即可。

        ht = HT_DIST()  # 从现在开始至下一次痊愈需等待的时间

        next_events[node] = (node, 'become S', time + ht)  # 记录下一次痊愈的事件。

        # —————————————————————————————————————对该节点邻居进行感染判定——————————————————————————————————————————
        for neighbor in G.neighbors(node):
            it = EXP(beta)  # 从现在开始至下一次节点node进行传染行为的等待时间

            if it < ht and states[neighbor] == 0:  # 若传染时间过长则节点无法在此次感染时间内传染该邻居
                next_events[neighbor] = (neighbor, 'become I', min(time + it, next_events[neighbor][2]))
                # 选取最近的时间点作为新的下一次的感染时间，将原来的事件覆盖
        # ———————————————————————————————————————————————————————————————————————————————————————————————————

    def become_S(node, time):
        """
        节点的痊愈函数，实现节点痊愈的效果。
        :param node: 节点的序号。
        :param time: 节点的痊愈时间。
        :return: 无
        """
        states[node] = 0  # 将该节点状态变为易感态
        if time != 0:
            num_now[1] -= 1  # 当前感染节点数减一
            num_now[0] += 1

        infect_times[node] = 0  # 因为该节点痊愈，感染时间为0。

        # ————————————————————————————————因为痊愈，则需再次考虑下次的感染时间。————————————————————————————————
        IT = 999999999  # 若周围无法传染取而代之作为无限的代表的数字。
        for neighbor in G.neighbors(node):
            it = EXP(beta)  # 从现在开始至下一次该邻居进行传染行为的等待时间
            if states[neighbor] == 1 and time + it < next_events[neighbor][2]:
                IT = min(it, IT)  # 仍然选择最短时间点作为感染时间点
        next_events[node] = (node, 'become I', IT + time)  # 写入下次的感染事件。
        # ———————————————————————————————————————————————————————————————————————————————————————————————

    # ————————————————————————————————初始化状态数量及感染的标签———————————————————————————————————
    num_now[1] = Counter(states)[1]
    num_now[0] = N - num_now[1]
    for node in range(N):
        if states[node] == 1:
            next_events[node] = (node, 'become I', time - infect_times[node])
        else:
            next_events[node] = (node, 'become S', time)
        states[node] = 1 - states[node]  # 这里的翻转状态是为了接下来依事件标签处理系统状态时条件判定的方便。
    number_of_node_states[0].append(time)
    number_of_node_states[1].append(num_now[0])
    number_of_node_states[2].append(num_now[1])
    # —————————————————————————————————————————————————————————————————————————————————————————
    # ——————————————————————————————————————————开始直到到达指定模拟时间的SIS模拟—————————————————————————————————————————————
    while time < T:
        event_now = min(next_events, key=lambda x: x[2])  # 取出时间点最小的事件作为当前事件
        time = event_now[2]  # 使用该事件的发生时间作为当前时间
        # ————————————————————进行传染或痊愈的判定——————————————————————
        if event_now[1] == 'become S' and states[event_now[0]] == 1:
            become_S(event_now[0], event_now[2])
        if event_now[1] == 'become I' and states[event_now[0]] == 0:
            become_I(event_now[0], event_now[2])
        # ——————————————————————————————————————————————————————————
        number_of_node_states[0].append(time)
        number_of_node_states[1].append(num_now[0])
        number_of_node_states[2].append(num_now[1])  # 记录当前时间的状态数量
        # ————————————————————————————————打印当前状态—————————————————————————————————————
        # print('\r当前时间进度百分比：', round(time / T * 100, 2), '%|当前状态数量分布：', num_now, end='')
        # ———————————————————————————————————————————————————————————————————————————————
    # ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    return number_of_node_states, infect_times.tolist()

if __name__ == '__main__':
    # ———————————————————网络参数———————————————————
    N = 500  # 网络节点数量
    k = 10  # 度
    G = nx.random_regular_graph(k, N)  # 均匀随机图
    # —————————————————————————————————————————————
    # ————————————————————————————————————————使用不同参数模拟————————————————————————————————————————
    Betas = np.arange(0.2, 0.21, 0.05)
    init_infected_rate = 0.2
    simu_time = 50
    Mus = np.arange(0, 1, 0.05)
    for Mu in Mus:

        # ————————————————SIS模型参数—————————————————

        beta = 1
        # 指数分布
        mu = 1

        # 幂律分布
        alpha = 4
        tmin = 5

        # 均匀分布
        a = 10
        b = 50

        # 对数正态分布
        sigma = 0.75


        def HT_DIST():
            """
            heal time distribution，即痊愈时间分布，是节点
            从被感染开始到完全痊愈所消耗时间的分布。这里要求没有
            参数输入。
            :return: 满足某个分布的随机数。
            """
            return np.random.lognormal(Mu, sigma)


        # ———————————————————————————————————————————
        # ————————————————————————————————理论预测————————————————————————————————————
        AVG_k = sum(dict(nx.degree(G)).values()) / N  # 平均度
        pIInfty_of_lognormal = 1 - np.exp(-Mu - sigma ** 2 / 2) / (beta * AVG_k)

        pIInfty = pIInfty_of_lognormal  # 最终感染密度
        print('\n最终感染密度：', pIInfty, '/感染数量：', N * pIInfty)  # 打印预测结果
        # ———————————————————————————————————————————————————————————————————————————
        # ——————————————————————————————————多轮模拟取平均——————————————————————————————————
        avgs = 50  # 总轮数
        S = [[], [], []]  # 记录模拟中所有事件发生时节点的状态数量
        Times = []  # 记录所有轮次运行结束时节点的已感染时间。
        for avg in range(avgs):
            # ——————————————————打印轮次——————————————————
            print('\n———————————————————————————————分界线———————————————————————————————')
            print('当前位于第', avg + 1, '轮。', '当前Mu:', round(Mu, 2), '应感染：', round(N * pIInfty, 0))
            # ——————————————————————————————————————————
            # ————————————————初始状态———————————————————
            pI0 = init_infected_rate  # 初始感染密度
            states = np.zeros(N)
            for node in range(N):
                if random() < pI0:
                    states[node] = 1
            # ——————————————————————————————————————————
            # ————————————————————开始本轮模拟—————————————————————
            T = simu_time  # 模拟时间
            s, times = simulate_SIS(T, beta, HT_DIST, G, states)
            # ———————————————————————————————————————————————————
            # ————处理本轮结果————
            for i in range(3):
                S[i] += s[i]
            Times += times
            # ——————————————————
        S = np.array(sorted(np.array(S).T.tolist(), key=lambda x: x[0])).T


        def moving_average(arr, n):
            """
            将数组按照每n个元素一组进行均值处理。
            :param arr: 输入的列表
            :param n: 取均值的分组长度
            :return: 一个长度为len(arr)//n的均值列表。
            """
            return np.mean(arr[:len(arr) // n * n].reshape(-1, n), axis=1)


        s = [[], [], []]
        for i in range(3):
            s[i] = moving_average(S[i], avgs)
        name = str([N, k, round(beta, 2), Mu, sigma, avgs, init_infected_rate, simu_time]) + '.npy'
        np.save(r'result/Times/lognormal/ET/'+name, np.array(Times))
        # ————————————————————————————————————————————————————————————————————————————————
    # —————————————————————————————————————————————————————————————————————————————————————————————
