import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'plot_signal',
    'bode_diagram'
]


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


def bode_diagram(xt, xf, pf=None, h=0.001, tlim=[-1, 1], flim=[-1, 1]):
    # Evaluate time domain function
    tvec = np.arange(tlim[0], tlim[1]+h, h)
    response = xt(tvec)
    
    # Evaluate magnitude function
    fvec = np.arange(flim[0], flim[1]+h, h)
    spectrum = xf(fvec)
    mag = np.abs(spectrum)
    
    # Evaluate phase function
    if pf is None:
        if not np.iscomplexobj(spectrum):
            skip_phase = True
        else:
            phase = np.arctan2(spectrum.imag, spectrum.real)
            skip_phase = False
    else:
        phase = pf(fvec)
        skip_phase = False
    
    # Initialize subplot grid
    grid = plt.GridSpec(2, 2, wspace=0.4)
    fig = plt.figure()
    ax1 = fig.add_subplot(grid[:, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 1])
    
    # Plot time signal
    ax1.plot(tvec, response)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot magnitude spectrum
    ax2.plot(fvec, mag)
    ax2.set_ylabel('Magnitude [-]')
    ax2.grid(True)
    ax2.set_xticklabels([])
    
    # Plot phase
    if not skip_phase:
        ax3.plot(fvec, phase)
        ax3.set_xlabel('Frequency [Hz]')
        ax3.set_ylabel('Phase [rad]')
        ax3.grid(True)
        
    fig.align_labels()
    
    return fig, (ax1, ax2, ax3)


def parse_sympy_expr(expr, symbol):
    if isinstance(expr, (Add, Mul, Pow)):
        if symbol is None:
            raise ValueError('xt_sym must be a Sympy symbol.')
        return lambdify(symbol, expr)
    return expr