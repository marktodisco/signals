import numpy as np
import matplotlib.pyplot as plt
import math
from math import sin, cos, pi, inf

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
    'exp_pulse_2',
    'plot_signal'
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
    return sin(pi * t) / pi / t


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


def plot_signal(t, x, ret=False):
    fig, ax = plt.subplots(1, 1)
    
    # Plot axes lines
    opts = dict(c='k', lw=1.5)
    ax.plot([min(t), max(t)], [0, 0], **opts)
    ax.plot([0, 0], [min(x), max(x)], **opts)

    # Plot function
    ax.plot(t, x, 'b-', lw=3)
    fig.set_size_inches(10, 6)
    
    return fig if ret else None
