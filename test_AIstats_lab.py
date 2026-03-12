import numpy as np

from AI_stats_lab import (
    exponential_pdf,
    exponential_interval_probability,
    simulate_exponential_probability,
    gaussian_pdf,
    posterior_probability
)


# -------------------------------------
# Q1 – PDF correctness
# -------------------------------------

def test_exponential_pdf():
    val = exponential_pdf(1)
    expected = np.exp(-1)
    assert np.isclose(val, expected, atol=1e-6)


# -------------------------------------
# Q1 – Analytical probability
# -------------------------------------

def test_exponential_interval():
    expected = np.exp(-2) - np.exp(-5)
    result = exponential_interval_probability(2,5)
    assert np.isclose(result, expected, atol=1e-6)


# -------------------------------------
# Q1 – Simulation check
# -------------------------------------

def test_exponential_simulation():
    result = simulate_exponential_probability(2,5)
    expected = np.exp(-2) - np.exp(-5)
    assert abs(result - expected) < 0.02


# -------------------------------------
# Q2 – Gaussian PDF
# -------------------------------------

def test_gaussian_pdf():
    val = gaussian_pdf(40,40,2)
    expected = 1/(np.sqrt(2*np.pi)*2)
    assert np.isclose(val, expected, atol=1e-6)


# -------------------------------------
# Q2 – Posterior probability
# -------------------------------------

def test_posterior():
    result = posterior_probability(42)

    num = 0.7*np.exp(-(42-45)**2/4)
    den = 0.3*np.exp(-(42-40)**2/4) + num
    expected = num/den

    assert np.isclose(result, expected, atol=1e-3)
