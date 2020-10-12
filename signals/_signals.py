import numpy as np
import math

__all__ = [
    'u',
    'r',
    'sgn',
    'rect',
    'tri',
    'sinc',
    'dirac',
    'tri_pulse',
    'exp_pulse_1',
    'exp_pulse_2'
]


@np.vectorize
def u(t):
    return 0. if t < 0. else 1.


@np.vectorize
def r(t):
    return 0. if t < 0. else float(t)


@np.vectorize
def sgn(t):
    return -1. if t < 0. else 1.


@np.vectorize
def rect(t):
    return 1. if abs(t) < 0.5 else 0.


@np.vectorize
def tri(t):
    return 1. - abs(t) if abs(t) <= 1 else 0.


@np.vectorize
def sinc(t):
    return math.sin(math.pi * t) / math.pi / t


@np.vectorize
def dirac(t):
    return math.inf if t == 0. else 0.


@np.vectorize
def tri_pulse(t, tau):
    return tri(t / tau) / tau


@np.vectorize
def exp_pulse_1(t, tau):
    "math.exp(-t / tau) * u(t)"
    return math.exp(-t / tau) * u(t)


@np.vectorize
def exp_pulse_2(t, tau):
    "math.exp(-t / tau) * u(t)"
    return math.exp(-(abs(t) / tau)) * 2 / tau
