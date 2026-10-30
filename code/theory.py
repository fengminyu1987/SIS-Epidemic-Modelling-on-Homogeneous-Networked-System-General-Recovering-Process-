"""Mean-field predictions reported in the grp-SIS paper."""

from __future__ import annotations

import math
import numpy as np


def mean_recovery_time(distribution: str, **params: float) -> float:
    if distribution == "exponential":
        return 1.0 / params["mu"]
    if distribution == "lognormal":
        return math.exp(params["mu"] + 0.5 * params["sigma"] ** 2)
    if distribution == "powerlaw":
        alpha = params["alpha"]
        return (alpha - 2.0) / (alpha - 1.0) * params["t0"]
    raise ValueError(distribution)


def threshold(distribution: str, k: float, **params: float) -> float:
    return 1.0 / (k * mean_recovery_time(distribution, **params))


def endemic_prevalence(
    beta: np.ndarray | float, distribution: str, k: float, **params: float
) -> np.ndarray:
    betas = np.asarray(beta, dtype=float)
    return np.maximum(1.0 - threshold(distribution, k, **params) / betas, 0.0)


def recovery_pdf(x: np.ndarray, distribution: str, **params: float) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    if distribution == "exponential":
        return params["mu"] * np.exp(-params["mu"] * values)
    if distribution == "lognormal":
        mu, sigma = params["mu"], params["sigma"]
        result = np.zeros_like(values)
        positive = values > 0
        result[positive] = (
            np.exp(-((np.log(values[positive]) - mu) ** 2) / (2 * sigma**2))
            / (values[positive] * sigma * math.sqrt(2 * math.pi))
        )
        return result
    if distribution == "powerlaw":
        alpha, t0 = params["alpha"], params["t0"]
        result = np.zeros_like(values)
        active = values >= t0
        result[active] = (alpha - 1) / t0 * (values[active] / t0) ** (-alpha)
        return result
    raise ValueError(distribution)
