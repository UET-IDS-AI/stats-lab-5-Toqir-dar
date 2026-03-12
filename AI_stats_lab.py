import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    pass


def exponential_interval_probability(a, b, lam=1):
    """
    Compute P(a < X < b) using analytical formula.
    """
    pass


def simulate_exponential_probability(a, b, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    pass


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    pass


def posterior_probability(time):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    pass


def simulate_posterior_probability(time, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    pass
