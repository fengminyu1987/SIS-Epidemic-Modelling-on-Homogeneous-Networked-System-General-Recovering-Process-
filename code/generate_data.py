"""Generate the numeric data needed by the grp-SIS figures.

No simulation result is shipped in the repository.  Run one experiment at a
time, or use ``--experiment all`` for a complete reproduction.  The default
``paper`` profile matches the settings stated in the manuscript.  ``smoke`` is
a short end-to-end check suitable for CI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from metrics import exponential_kl, lognormal_kl, power_law_kl
from sis_model import (
    INITIAL_INFECTED_RATE,
    initial_states,
    make_random_regular_graph,
    recovery_sampler,
    simulate_sis,
)
from theory import endemic_prevalence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"

TEMPORAL_CASES = {
    "exponential": {"beta": 0.26, "mu": 0.5},
    "lognormal": {"beta": 0.33, "mu": 0.0, "sigma": 1.0},
    "powerlaw": {"beta": 0.30, "alpha": 4.0, "t0": 1.0},
}

PREVALENCE_GRIDS = {
    "exponential": np.linspace(0.02, 0.50, 25),
    "lognormal": np.linspace(0.03, 0.60, 20),
    "powerlaw": np.linspace(0.03, 0.66, 22),
}


def profile(name: str) -> dict:
    if name == "paper":
        return {
            "nodes": 2500,
            "runs": 50,
            "horizon": 50.0,
            "time_points": 600,
            "k_values": [4, 6, 10],
        }
    return {
        "nodes": 200,
        "runs": 4,
        "horizon": 8.0,
        "time_points": 180,
        "k_values": [4, 6, 10],
    }


def _run(
    n: int,
    k: int,
    horizon: float,
    beta: float,
    distribution: str,
    params: dict,
    seed: int,
):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    graph = make_random_regular_graph(n, k, seed)
    states = initial_states(n, rng, INITIAL_INFECTED_RATE)
    sampler = recovery_sampler(distribution, **params)
    records, ages = simulate_sis(horizon, beta, sampler, graph, states)
    return records, np.asarray(ages, dtype=float)


def _resample(records, grid: np.ndarray) -> np.ndarray:
    times = np.asarray(records[0], dtype=float)
    infected = np.asarray(records[2], dtype=float)
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, len(infected) - 1)
    return infected[indices]


def temporal(args, data_dir: Path) -> None:
    settings = profile(args.profile)
    n = args.nodes or settings["nodes"]
    runs = args.runs or settings["runs"]
    horizon = settings["horizon"]
    points = settings["time_points"]
    grid = np.concatenate(([0.0], np.geomspace(0.01, horizon, points)))
    result = {"time": grid}
    ages_by_distribution = {}
    table = {}
    started = time.time()

    for name, case in TEMPORAL_CASES.items():
        params = {key: value for key, value in case.items() if key != "beta"}
        beta = case["beta"]
        trajectories = []
        ages = []
        for run_index in range(runs):
            records, final_ages = _run(
                n, 10, horizon, beta, name, params, args.seed + run_index
            )
            trajectories.append(_resample(records, grid) / n)
            ages.extend(final_ages[final_ages > 0].tolist())
        values = np.asarray(trajectories)
        result[f"{name}_runs"] = values
        result[f"{name}_mean"] = values.mean(axis=0)
        result[f"{name}_low"] = np.percentile(values, 2.5, axis=0)
        result[f"{name}_high"] = np.percentile(values, 97.5, axis=0)
        result[f"{name}_steady"] = endemic_prevalence(beta, name, 10, **params)
        ages_by_distribution[name] = np.asarray(ages)

        if name == "exponential":
            kl = exponential_kl(ages_by_distribution[name], params["mu"])
        elif name == "lognormal":
            kl = lognormal_kl(
                ages_by_distribution[name], params["mu"], params["sigma"]
            )
        else:
            kl = power_law_kl(
                ages_by_distribution[name], params["alpha"], params["t0"]
            )
        table[name] = {"parameters": params, "beta": beta, "kl": kl}
        print(f"[temporal] {name}: {runs} runs finished")

    np.savez_compressed(data_dir / "fig3_temporal.npz", **result)
    np.savez_compressed(
        data_dir / "fig4_confidence.npz",
        time=grid,
        **{
            f"{name}_{key}": value
            for name in TEMPORAL_CASES
            for key, value in (
                ("mean", result[f"{name}_mean"]),
                ("low", result[f"{name}_low"]),
                ("high", result[f"{name}_high"]),
                ("std", result[f"{name}_runs"].std(axis=0)),
            )
        },
    )
    np.savez_compressed(data_dir / "fig6_ages.npz", **ages_by_distribution)
    (data_dir / "table2_kl.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[temporal] saved in {time.time() - started:.1f}s")


def _prevalence_grid(args, short_horizon: bool, data_dir: Path) -> None:
    settings = profile(args.profile)
    n = args.nodes or settings["nodes"]
    runs = args.runs or settings["runs"]
    horizon = 10.0 if short_horizon else settings["horizon"]
    payload = {}
    for name, full_grid in PREVALENCE_GRIDS.items():
        grid = full_grid if args.profile == "paper" else np.linspace(
            full_grid[0], full_grid[-1], 7
        )
        params = {key: value for key, value in TEMPORAL_CASES[name].items() if key != "beta"}
        for k in settings["k_values"]:
            means = []
            low = []
            high = []
            for beta in grid:
                values = []
                for run_index in range(runs):
                    records, _ = _run(
                        n,
                        k,
                        horizon,
                        float(beta),
                        name,
                        params,
                        args.seed + 10_000 * k + run_index,
                    )
                    final_i = np.mean(records[2][-5:]) / n if len(records[2]) >= 5 else 0.0
                    values.append(final_i)
                values_array = np.asarray(values)
                means.append(values_array.mean())
                low.append(np.percentile(values_array, 16))
                high.append(np.percentile(values_array, 84))
            prefix = f"{name}_k{k}"
            payload[f"{prefix}_beta"] = grid
            payload[f"{prefix}_mean"] = np.asarray(means)
            payload[f"{prefix}_low"] = np.asarray(low)
            payload[f"{prefix}_high"] = np.asarray(high)
            print(f"[prevalence{'10' if short_horizon else ''}] {name}, k={k} finished")
    filename = "fig5d_errors.npz" if short_horizon else "fig5_prevalence.npz"
    np.savez_compressed(data_dir / filename, **payload)


def prevalence(args, data_dir: Path) -> None:
    _prevalence_grid(args, False, data_dir)
    if not args.skip_short:
        _prevalence_grid(args, True, data_dir)


def et_lognormal(args, data_dir: Path) -> None:
    settings = profile(args.profile)
    n = args.nodes or settings["nodes"]
    runs = args.runs or settings["runs"]
    horizon = settings["horizon"]
    mus = (
        np.linspace(0.0, 0.95, 20)
        if args.profile == "paper"
        else np.linspace(0.0, 0.8, 5)
    )
    sigmas = [0.5, 1.0, 1.5]
    means = np.zeros((len(sigmas), len(mus)))
    sems = np.zeros_like(means)
    for row, sigma in enumerate(sigmas):
        for col, mu in enumerate(mus):
            samples = []
            for run_index in range(runs):
                _, ages = _run(
                    n,
                    10,
                    horizon,
                    1.0,
                    "lognormal",
                    {"mu": float(mu), "sigma": sigma},
                    args.seed + 10_000 * row + run_index,
                )
                positive = ages[ages > 0]
                if len(positive):
                    samples.append(positive.mean())
            means[row, col] = np.mean(samples)
            sems[row, col] = np.std(samples, ddof=1) / math.sqrt(len(samples)) if len(samples) > 1 else 0
        print(f"[fig7a] sigma={sigma} finished")
    np.savez_compressed(
        data_dir / "fig7_lognormal.npz",
        mu=mus,
        sigma=np.asarray(sigmas),
        mean=means,
        sem=sems,
    )


def et_powerlaw(args, data_dir: Path) -> None:
    settings = profile(args.profile)
    n = args.nodes or settings["nodes"]
    runs = args.runs or settings["runs"]
    horizon = settings["horizon"]
    alphas = (
        np.linspace(4.0, 5.0, 26)
        if args.profile == "paper"
        else np.linspace(4.0, 5.0, 6)
    )
    t0_values = [1.0, 2.0, 5.0]
    means = np.zeros((len(t0_values), len(alphas)))
    sems = np.zeros_like(means)
    for row, t0 in enumerate(t0_values):
        for col, alpha in enumerate(alphas):
            samples = []
            for run_index in range(runs):
                _, ages = _run(
                    n,
                    10,
                    horizon,
                    1.0,
                    "powerlaw",
                    {"alpha": float(alpha), "t0": t0},
                    args.seed + 10_000 * row + run_index,
                )
                positive = ages[ages > 0]
                if len(positive):
                    samples.append(positive.mean())
            means[row, col] = np.mean(samples)
            sems[row, col] = np.std(samples, ddof=1) / math.sqrt(len(samples)) if len(samples) > 1 else 0
        print(f"[fig7b] t0={t0} finished")
    np.savez_compressed(
        data_dir / "fig7_powerlaw.npz",
        alpha=alphas,
        t0=np.asarray(t0_values),
        mean=means,
        sem=sems,
    )


def write_manifest(args, data_dir: Path) -> None:
    manifest = {
        "profile": args.profile,
        "seed": args.seed,
        "nodes_override": args.nodes,
        "runs_override": args.runs,
        "temporal_cases": TEMPORAL_CASES,
        "initial_infected_rate": INITIAL_INFECTED_RATE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["temporal", "prevalence", "et-lognormal", "et-powerlaw", "all"],
        required=True,
    )
    parser.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    parser.add_argument("--seed", type=int, default=20250326)
    parser.add_argument("--nodes", type=int, help="Override N (normally 2500)")
    parser.add_argument("--runs", type=int, help="Override the number of replicates")
    parser.add_argument("--skip-short", action="store_true", help="Skip the T=10 prevalence run")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    actions: dict[str, Callable[[argparse.Namespace, Path], None]] = {
        "temporal": temporal,
        "prevalence": prevalence,
        "et-lognormal": et_lognormal,
        "et-powerlaw": et_powerlaw,
    }
    if args.experiment == "all":
        for action in actions.values():
            action(args, args.output)
    else:
        actions[args.experiment](args, args.output)
    write_manifest(args, args.output)


if __name__ == "__main__":
    main()
