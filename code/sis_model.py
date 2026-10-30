"""Exact event-driven simulator used by the grp-SIS experiments.

The first implementation in ``simulate_sis_reference`` mirrors the author's
original ``main.py`` line by line.  ``simulate_sis`` keeps the same event
semantics but replaces the O(N) minimum scan with a lazy binary heap, making a
full reproduction run practical.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import Counter
from typing import Callable, Tuple

import networkx as nx
import numpy as np


INF = 999_999_999.0
INITIAL_INFECTED_RATE = 0.30


def power_law_sample(alpha: float, t0: float) -> float:
    """Match ``HT_DIST_powerlaw`` in the original scripts."""
    value = INF
    while value > 10_000:
        u = random.random()
        value = t0 * math.pow(u, 1.0 / (1.0 - alpha))
    return value


def recovery_sampler(distribution: str, **params: float) -> Callable[[], float]:
    """Return a no-argument recovery-time sampler."""
    if distribution == "exponential":
        return lambda: random.expovariate(params["mu"])
    if distribution == "lognormal":
        return lambda: random.lognormvariate(params["mu"], params["sigma"])
    if distribution == "powerlaw":
        return lambda: power_law_sample(params["alpha"], params["t0"])
    raise ValueError(f"Unknown recovery distribution: {distribution}")


def simulate_sis_reference(
    T: float,
    beta: float,
    recovery_time: Callable[[], float],
    graph: nx.Graph,
    initial_states: np.ndarray,
) -> Tuple[list[list[float]], list[float]]:
    """Reference implementation retained for direct comparison with ``main.py``."""
    states = np.asarray(initial_states, dtype=float).copy()
    n = len(states)
    events = [(node, "", INF) for node in range(n)]
    infect_times = np.zeros(n)
    records: list[list[float]] = [[], [], []]
    counts = [0, 0]
    now = 0.0

    def become_i(node: int, time: float) -> None:
        states[node] = 1
        if time != 0:
            counts[1] += 1
            counts[0] -= 1
        infect_times[node] = T - time
        heal_time = recovery_time()
        events[node] = (node, "become S", time + heal_time)
        for neighbor in graph.neighbors(node):
            infection_wait = random.expovariate(beta)
            if infection_wait < heal_time and states[neighbor] == 0:
                events[neighbor] = (
                    neighbor,
                    "become I",
                    min(time + infection_wait, events[neighbor][2]),
                )

    def become_s(node: int, time: float) -> None:
        states[node] = 0
        if time != 0:
            counts[1] -= 1
            counts[0] += 1
        infect_times[node] = 0
        next_infection = INF
        for neighbor in graph.neighbors(node):
            infection_wait = random.expovariate(beta)
            if states[neighbor] == 1 and time + infection_wait < events[neighbor][2]:
                next_infection = min(infection_wait, next_infection)
        events[node] = (node, "become I", next_infection + time)

    counts[1] = Counter(states)[1]
    counts[0] = n - counts[1]
    for node in range(n):
        if states[node] == 1:
            events[node] = (node, "become I", now)
        else:
            events[node] = (node, "become S", now)
        states[node] = 1 - states[node]
    records[0].append(now)
    records[1].append(counts[0])
    records[2].append(counts[1])

    while now < T:
        event = min(events, key=lambda item: item[2])
        now = event[2]
        node, kind = event[0], event[1]
        if kind == "become S" and states[node] == 1:
            become_s(node, event[2])
        if kind == "become I" and states[node] == 0:
            become_i(node, event[2])
        records[0].append(now)
        records[1].append(counts[0])
        records[2].append(counts[1])
    return records, infect_times.tolist()


def simulate_sis(
    T: float,
    beta: float,
    recovery_time: Callable[[], float],
    graph: nx.Graph,
    initial_states: np.ndarray,
) -> Tuple[list[list[float]], list[float]]:
    """Event-driven grp-SIS simulation with the same update order as the paper code."""
    states = np.asarray(initial_states, dtype=float).copy()
    n = len(states)
    events = [(node, "", INF) for node in range(n)]
    versions = [0] * n
    heap: list[tuple[float, int, str, int]] = []
    infect_times = np.zeros(n)
    records: list[list[float]] = [[], [], []]
    counts = [0, 0]
    now = 0.0

    def schedule(node: int, kind: str, time: float) -> None:
        events[node] = (node, kind, time)
        versions[node] += 1
        heapq.heappush(heap, (time, node, kind, versions[node]))

    def become_i(node: int, time: float) -> None:
        states[node] = 1
        if time != 0:
            counts[1] += 1
            counts[0] -= 1
        infect_times[node] = T - time
        heal_time = recovery_time()
        schedule(node, "become S", time + heal_time)
        for neighbor in graph.neighbors(node):
            infection_wait = random.expovariate(beta)
            if infection_wait < heal_time and states[neighbor] == 0:
                candidate = time + infection_wait
                if candidate < events[neighbor][2]:
                    schedule(neighbor, "become I", candidate)

    def become_s(node: int, time: float) -> None:
        states[node] = 0
        if time != 0:
            counts[1] -= 1
            counts[0] += 1
        infect_times[node] = 0
        next_infection = INF
        for neighbor in graph.neighbors(node):
            infection_wait = random.expovariate(beta)
            if states[neighbor] == 1 and time + infection_wait < events[neighbor][2]:
                next_infection = min(infection_wait, next_infection)
        schedule(node, "become I", time + next_infection)

    counts[1] = Counter(states)[1]
    counts[0] = n - counts[1]
    for node in range(n):
        if states[node] == 1:
            schedule(node, "become I", now)
        else:
            schedule(node, "become S", now)
        states[node] = 1 - states[node]
    records[0].append(now)
    records[1].append(counts[0])
    records[2].append(counts[1])

    while now < T:
        time, node, kind, version = heapq.heappop(heap)
        if version != versions[node]:
            continue
        now = time
        if kind == "become S" and states[node] == 1:
            become_s(node, time)
        if kind == "become I" and states[node] == 0:
            become_i(node, time)
        records[0].append(now)
        records[1].append(counts[0])
        records[2].append(counts[1])
    return records, infect_times.tolist()


def make_random_regular_graph(n: int, k: int, seed: int) -> nx.Graph:
    return nx.random_regular_graph(k, n, seed=seed)


def initial_states(n: int, rng: np.random.Generator, infected_rate: float) -> np.ndarray:
    return (rng.random(n) < infected_rate).astype(float)

