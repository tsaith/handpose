import numpy as np
import matplotlib.pyplot as plt
from .stft import stft, istft


def fourier_spectrum(y, dt, spectrum_type='power'):
    """
    Fourier spectrum.
    
    Parameters
    ----------
    y: array-like
        Input signals.
    
    dt: float
        Sampling interval (in seconds).
    
    spectrum_type: str
        Spectrum type; 
        'power' for power spectrum, 
        'amplitude' for amplitude spectrum. 
    
    Returns
    -------
    spectrum: ndarray
        Spectrum array.
        
    freq: ndarray
        Frequency array. 

    """
    
    num_samples = len(y)
    num_half = num_samples / 2
    
    Y = np.fft.fft(y) # FFT
    Y /= num_samples  # Normalized
    
    # Frequency array
    freq = np.fft.fftfreq(num_samples, dt) 
    
    amp = np.abs(Y)  # Amplitude spectrum
    power = amp*amp  # Power spectrum
    
    indices = range(num_half)
    freq = freq[indices]
    
    amp = amp[indices]
    power = power[indices]
    # Power or amplitude spectrum
    spectrum = {'power': power, 'amplitude': amp}.get(spectrum_type)

    return spectrum, freq
    

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

    return Z, X, Y 


def stft_tf_plot(sig, t, f_min=None, f_max=None, 
                 frame_size=None, hop=None, yscale='log'):
    """
    Time-frequency plot based on Short-Time Fourier transform.
    
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
    frame_size: float
        Frame size in seconds.
    hop: float
        Time shift between two frames in seconds. 
    yscale: str
        Scale of the y axis; 'log' or 'linear'.
    Returns
    -------
        Z: ndarray
            Contour array of plot.
        X: ndarray
            X grids of plot.
        Y: ndarray
            Y grids of plot.
    """

    num_samples = len(sig) # Number of time samples
    dt = t[1] - t[0] # Time interval of sampling
    
    fs = 1.0 / dt # Sampling rate
    total_time = dt*num_samples 
        
    if frame_size == None:
        frame_size = 0.1 * total_time 
    if hop == None:
        hop = 0.01*total_time   
    
    # Frequency resolution
    df = 1.0 / frame_size

    # Short-Time Fourier transform
    W = stft(sig, fs, frame_size, hop) 
    W /= (frame_size * fs)

    # Power
    amp = np.abs(W)
    power = amp*amp
    

    Fz = int(0.5 * frame_size * fs)
    f_nyquist = (Fz-1)*df # Nyquist frequency
    
    # Plot time-frequency relation
    t_min = t[0]
    t_max = t[-1]
    
    if f_min == None:
        f_min = 1
    if f_max == None:    
        f_max = f_nyquist

    
    fig = plt.figure()    
    ax = fig.add_subplot(1, 1, 1)
    Z = power[:, :Fz].T # Z is of [frquency, time] format
    X, Y = np.meshgrid(np.linspace(t_min, t_max, Z.shape[1]), 
                       np.linspace(f_min, f_nyquist, Z.shape[0]))
    
    pcm = ax.pcolormesh(X, Y, Z, cmap='RdBu_r')
    ax.set_title("Power")
    ax.set_xlabel("Time (second)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(t_min, t_max)
    ax.set_yscale(yscale)
    ax.set_ylim(f_min, f_max)
    fig.colorbar(pcm, ax=ax, extend='both', orientation='vertical')
    
    return Z, X, Y

