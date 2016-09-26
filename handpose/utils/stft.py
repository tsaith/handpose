import numpy as np


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

