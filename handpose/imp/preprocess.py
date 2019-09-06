import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from ..utils import csv2numpy

def ri_to_magnitude(arr, with_phase=True):
    """
    Convert the real and imaginary parts into magnitude.
    
    Parameters
    ----------
    arr: array
        Input array.
    with_pahse: bool
         calculate the phase or not.
        
    Returns
    -------
        mag: array
            Magnitudes.
        phases: array
            Phase angles
    """
    
    eps_tol = 1e-8 # Error of tolerance 
    
    reals = arr[:, 0::2] # Real part
    imags = arr[:, 1::2] + eps_tol # Imaginary part
    
    # Magnitudes
    magnitudes = np.sqrt(reals*reals + imags*imags)
    
    if with_phase: # Find phase angles
        phases = np.arctan2(imags, reals)
        out = (magnitudes, phases)
    else:    
        out = magnitudes
    
    return out     

def generate_instance_file(f, fc, cols=28*2, keyword='_rec_'):
    """
    Generate the instance file.
    
    Parameters
    ----------
    f: str
        Path of the target file.
    fc: str
        Path to the calibration file.
    cols: int
        Number of columns to be kept.
        
    """
    
    dir_path = os.path.dirname(f)
    filename = os.path.basename(f)
    dir_out = "output"
    filename_out = filename.replace(keyword, "_")
  
    # Create the output directory if it does not exit
    dir_out_path = os.path.join(dir_path, dir_out)
    if not os.path.exists(dir_out_path):
        os.makedirs(dir_out_path)
    
    # Load data files
    ri   = csv2numpy(f, start_col=1)
    ri_c = csv2numpy(fc, start_col=1)

    # Only keep necessary columns 
    ri = ri[:, :cols]
    ri_c = ri_c[:, :cols]
        
    # Magnitudes and angles    
    mag, phase = ri_to_magnitude(ri)
    mag0, phase0 = ri_to_magnitude(ri_c)
    
    mag0_base = mag0[0, :]
    phase0_base = phase0[0, :]

    rows = mag.shape[0]
    mag0 = np.zeros((rows, cols/2))
    phase0 = np.zeros((rows, cols/2))
    for i in range(rows):
        mag0[i,:] = mag0_base
        phase0[i,:] = phase0_base
    
    # Difference of magnitudes and phases
    dmag = mag - mag0
    dphase = phase - phase0
    
    # Write to a file
    instance = np.concatenate((mag0, phase0, dmag, dphase), axis=1)
    fout = os.path.join(dir_out_path, filename_out)
    np.savetxt(fout, instance, delimiter=",")
    
    return instance
    
