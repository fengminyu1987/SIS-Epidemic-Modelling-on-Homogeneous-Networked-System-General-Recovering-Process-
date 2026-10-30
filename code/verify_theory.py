"""Fast, deterministic checks that do not require a full Monte Carlo run."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

import sis_model
from theory import endemic_prevalence, mean_recovery_time, threshold


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    graph = sis_model.make_random_regular_graph(100, 4, 42)
    states = np.zeros(100)
    states[:20] = 1
    cases = {
        "exponential": {"mu": 0.5},
        "lognormal": {"mu": 0.0, "sigma": 1.0},
        "powerlaw": {"alpha": 4.0, "t0": 1.0},
    }
    exact_events = True
    exact_ages = True
    equivalence = {}
    for distribution, params in cases.items():
        random.seed(123)
        reference = sis_model.simulate_sis_reference(
            10,
            0.2,
            sis_model.recovery_sampler(distribution, **params),
            graph,
            states,
        )
        random.seed(123)
        optimized = sis_model.simulate_sis(
            10,
            0.2,
            sis_model.recovery_sampler(distribution, **params),
            graph,
            states,
        )
        same_events = np.array_equal(np.asarray(reference[0]), np.asarray(optimized[0]))
        same_ages = np.allclose(reference[1], optimized[1])
        equivalence[distribution] = {
            "events_equal": bool(same_events),
            "final_ages_equal": bool(same_ages),
        }
        exact_events = exact_events and same_events
        exact_ages = exact_ages and same_ages
    checks = {
        "fast_vs_reference_exact_events": bool(exact_events),
        "fast_vs_reference_exact_final_ages": bool(exact_ages),
        "equivalence_by_distribution": equivalence,
        "exponential_mean_recovery": mean_recovery_time("exponential", mu=0.5),
        "exponential_threshold_k10": threshold("exponential", 10, mu=0.5),
        "powerlaw_mean_recovery": mean_recovery_time("powerlaw", alpha=4, t0=1),
        "powerlaw_threshold_k10": threshold("powerlaw", 10, alpha=4, t0=1),
        "prevalence_at_beta_0_5_k10_exp": float(
            endemic_prevalence(0.5, "exponential", 10, mu=0.5)
        ),
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "verification.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not exact_events or not exact_ages:
        raise SystemExit("Optimized simulator differs from reference implementation")


if __name__ == "__main__":
    main()

