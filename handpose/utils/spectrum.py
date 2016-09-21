import numpy as np

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
    
    
