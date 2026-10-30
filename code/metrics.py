"""Distribution distances used for the KL table."""

from __future__ import annotations

from math import erfc, exp, log, sqrt

import numpy as np
from scipy.stats import entropy, expon


def lognormal_age_pdf(bin_centers, mu, sigma):
    result = np.zeros(len(bin_centers))
    for i, value in enumerate(bin_centers):
        if value > 0:
            result[i] = erfc((log(value) - mu) / (sqrt(2) * sigma)) / (
                2 * exp(mu + sigma**2 / 2)
            )
        elif value == 0:
            result[i] = exp(-mu - sigma**2 / 2)
    return result


def power_law_age_pdf(bin_centers, alpha, tmin):
    result = np.zeros(len(bin_centers))
    for i, value in enumerate(bin_centers):
        result[i] = (alpha - 2) / ((alpha - 1) * tmin)
        if value > tmin:
            result[i] *= (value / tmin) ** (1 - alpha)
        if value < 0:
            result[i] *= 0
    return result


def histogram_kl(samples, left, right, theoretical_pdf, num_bins=50):
    """Use the same 50-bin estimator as the author's KL scripts."""
    del left, right
    hist, edges = np.histogram(samples, bins=num_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(entropy(hist, theoretical_pdf(centers)))


def exponential_kl(samples, mu, num_bins=50):
    return histogram_kl(
        samples,
        0,
        np.max(samples),
        lambda centers: expon.pdf(centers, scale=1 / mu),
        num_bins,
    )


def lognormal_kl(samples, mu, sigma, num_bins=50):
    return histogram_kl(
        samples,
        0,
        np.max(samples),
        lambda centers: lognormal_age_pdf(centers, mu, sigma),
        num_bins,
    )


def power_law_kl(samples, alpha, tmin, num_bins=50):
    return histogram_kl(
        samples,
        0,
        np.max(samples),
        lambda centers: power_law_age_pdf(centers, alpha, tmin),
        num_bins,
    )

