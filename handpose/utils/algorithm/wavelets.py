import numpy as np
import pywt
import matplotlib.pyplot as plt

def cwt(data, fs, wavelet='mexh'):
    """
    Perform the continuous wavelet transform on data.

    Parameters
    ----------
    data: array
        1D data.
    fs: float
        Sampling frequency.
    wavelet: str
        Name of wavelet. Candidates: 'mexh' and 'cmor'.

    Returns
    -------
    cfs: array
        Coefficients in shape of (freq, times).
    freq: array
        Freqyency array.
    """
    scale_max = int(fs/4)
    if scale_max < 128:
        scale_max = 128
    scales = np.arange(1, scale_max)
    dt = 1.0/fs
    [cfs, freq] = pywt.cwt(data, scales, wavelet, dt)

    return cfs, freq

def cwt_plot(cfs, time, freq, xlabel=None, ylabel=None, yscale='log'):
    """
    Make CWT plot.

    Parameters
    ----------
    cfs: array
        Coefficients in shape of (freq, times).
    freq: array
        Freqyency array.
    xlabel: str
        x-axis label.
    ylabel: str
        y-axis label.
    yscale: str
        y-axis scale type. Candicateds: 'log' and 'linear'

    Returns
    -------
    fig: object
        Figure object.
    ax: object
        Axis object.
    """

    Z = abs(cfs) ** 2 # Power
    fig, ax = plt.subplots()
    T, F = np.meshgrid(time, freq)
    pcm = ax.pcolormesh(T, F, Z, cmap='RdBu_r')
    ax.set_title("Power")
    if xlabel is None:
        ax.set_xlabel("Time (second)")
    else:
        ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel("Frequency (Hz)")
    else:
        ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')

    return fig, ax

def cwt_tf_plot(sig, t, f_min=1, f_max=1000, yscale='log', tradeoff=30):
    """
    Time-frequency plot based on continuous wavelet transform.

    Parameters
    ----------
    sig: ndarray
        Signal array.
    t: ndarray
        Time array
    f_min: float
        Minimum frequency.
    f_max: float
        Maximun frequency.   
    yscale: str
        Scale of the y axis; 'log' or 'linear'.
    tradeoff: float     
        parameter for the wavelet, 
        tradeoff between time and frequency resolution.
    Returns
    -------
        Z: ndarray
            Contour array of plot.
        X: ndarray
            X grids of plot.
        Y: ndarray
            Y grids of plot.
    """

    from obspy.signal.tf_misfit import cwt

    dt = t[1] - t[0] # Time interval

    # Wavelet transform
    W = cwt(sig, dt, tradeoff, f_min, f_max)

    amp = np.abs(W) # Amplitude
    power = amp*amp # Power

    # Plot a figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    Z = power
    X, Y = np.meshgrid(t, 
        np.logspace(np.log10(f_min), np.log10(f_max), Z.shape[0]))
    pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r')

    ax.set_title("Power")
    ax.set_xlabel("Time (second)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yscale(yscale)
    ax.set_ylim(f_min, f_max)
    # Colorbar
    fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')

    return fig, ax

