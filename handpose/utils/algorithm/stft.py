import numpy as np
import matplotlib.pyplot as plt

def stft(x, fs, frame_size, hop):
  """
  Perform STFT (Short-Time Fourier Transform).

  x: Input data.
  fs: Sampling rate.
  frame_size: Frame size.
  hop: Hop size
  """

  frame_samp = int(frame_size*fs)
  hop_samp = int(hop*fs)
  w = np.hanning(frame_samp) # Hanning window
  X = np.array([np.fft.fft(w*x[i:i+frame_samp])
               for i in range(0, len(x)-frame_samp, hop_samp)])
  return X

def istft(X, fs, T, hop):
  """
  Perform inverse STFT (Short-Time Fourier Transform).

  X: Input data.
  fs: Sampling rate.
  T: Total time duration.
  hop: Hop size.
  """

  x = np.zeros(T*fs)
  frame_samp = X.shape[1]
  hop_samp = int(hop*fs)

  for n,i in enumerate(range(0, len(x)-frame_samp, hop_samp)):
    x[i:i+frame_samp] += np.real(np.fft.ifft(X[n]))

  return x

def stft_plot(sig, t, f_min=None, f_max=None,
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

    return fig, ax

