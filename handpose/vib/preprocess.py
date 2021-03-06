import numpy as np
import pandas as pd
import os
import random
from ..utils import csv2numpy, list_files, fourier_spectrum

def accel_abs(accel):

    ax = accel[:,0]
    ay = accel[:,1]
    az = accel[:,2]

    a_abs = np.sqrt(ax*ax + ay*ay + az*az)

    return a_abs

def norm_3d(data):

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    return np.sqrt(x*x + y*y + z*z)

def find_seg_indexes(data, size=None, peak_ratio=0.25, verbose=0):
    """
    Find the indexes of segmentations.

    data: array
        1D array.

    peak_ratio: float
        Signal-to-noise ratio.

    """

    num_samples = len(data) # Number of samples

    # Move back to set the left bound.
    back = int(size/4)

    # When static, the magnitude of acceleration is one
    mag_static = 1.0

    # Absolute difference
    diff_abs = np.abs(data) - mag_static

    # Determining the threshold
    threshold = peak_ratio
    if verbose > 0:
        print("---- threshod = {}".format(threshold))

    # Start and end index of the segmentations
    ia_indexes = []
    iz_indexes = []

    scan_start = back
    scan_end = num_samples - int(0.5*size)
    i = scan_start
    while i < scan_end:
        i += 1
        if diff_abs[i] > threshold:
            ia_indexes.append(i-back)
            i += size - back

    for e in ia_indexes:
        iz_indexes.append(e + size)

    return ia_indexes, iz_indexes


def get_train_data(data_raw, ia_indexes, iz_indexes):

    num_samples = len(ia_indexes)
    size = iz_indexes[0] - ia_indexes[0]
    data = np.zeros((num_samples, size))

    i = 0
    for ia, iz in zip (ia_indexes, iz_indexes):
        data[i, :] = data_raw[ia:iz]
        i += 1
    
    return data


def vib_file_factory(dir_path, keyword="_rec_",
                     dof=6, seg_size=None, peak_ratio=0.25, verbose=0):
    """
    File factory of vibrational data.
    """

    files = list_files(dir_path=dir_path, keyword=keyword)
    num_files = len(files)

    dir_out = "output"

    # Dictionary of segmentations
    # {file_name: seg_counts}
    seg_dict = {}

    print("Start to generate files under the {} direcory...".format(dir_out))
    for f in files:

        # Load raw data
        fpath = os.path.join(dir_path, f)
        if verbose > 0:
            print("...loading {}".format(fpath))
        accel = csv2numpy(fpath, start_col=1)
        # Find the indexes of segmentations
        ax = accel[:,0]
        ay = accel[:,1]
        az = accel[:,2]
        a_abs = np.sqrt(ax*ax + ay*ay + az*az)
        ia_indexes, iz_indexes = find_seg_indexes(a_abs, size=seg_size,
            peak_ratio=peak_ratio, verbose=verbose)

        # check if accel array size is less than seg_indexes, then complement by the last data.
        if accel.shape[0] < iz_indexes[-1]:
            print("Warning: seg index size {} is bigger than raw data size {}, then complement by the last data.".format(iz_indexes[-1], accel.shape[0]))
            fill_rows = iz_indexes[-1] - accel.shape[0];
            for i in range(fill_rows):
                accel = np.vstack((accel, accel[-1]))
 
        # Prepare the output data
        num_rows = len(ia_indexes)
        num_cols = dof*seg_size
        data = np.zeros((num_rows, num_cols))
        i = 0
        for ia, iz in zip (ia_indexes, iz_indexes):
            data[i, :] = np.hstack((accel[ia:iz]))
            i += 1

        # Generate output file
        dir_path_out = os.path.join(dir_path, dir_out)
        if not os.path.exists(dir_path_out):
            os.makedirs(dir_path_out)
        fname_out = f.replace(keyword, "_")
        fpath_out = os.path.join(dir_path_out, fname_out)
        np.savetxt(fpath_out, data, delimiter=",")

        if verbose > 0:
            print("Create file: {}".format(fname_out))
            print("data shape: {}".format(data.shape))

        # Store segmentation information
        seg_dict[f] = num_rows

    print("There are {} files generated.".format(num_files))

    return seg_dict

def to_magnitude(data):
    """
    Convert to magnitude from 3D components.
    """

    num_dims = 3
    rows, cols_in = data.shape
    cols = cols_in / num_dims

    arr = np.zeros((rows, cols, num_dims))
    out = np.zeros((rows, cols))

    for i in range(rows):
        for k in range(num_dims):
            ia = k*cols
            iz = ia + cols
            arr[i, :, k] = data[i, ia:iz]

        out[i,:] = np.sqrt(arr[i, :, 0]**2 + arr[i, :, 1]**2 + arr[i, :, 2]**2)

    return out

def to_ts_format(data, dof=3):
    """
    Convert into time-series format.
    """

    num_features = len(data)
    assert num_features % dof == 0
    num_tdata = int(num_features / dof)

    out = np.zeros((num_tdata, dof))
    for d in range(dof):
        ja = d*num_tdata
        jz = ja+ num_tdata
        out[:, d] = data[ja:jz]

    return out

def time_shift(data, backward=-10, forward=30):
    """
    Random shift the data's time dimension.
    """
    return np.roll(data, random.randint(backward, forward), axis=0)


def roll_axis(data, shift, axis=None):
    """
    Roll the data about the specific axis.
    """
    num_samples = len(data)
    num_dims = 3
    fs = num_samples / num_dims

    out = data.copy()

    for d in range(num_dims):
        ia = d*fs
        iz = ia + fs
        out[:, ia:iz] = np.roll(data[:, ia:iz], shift, axis=axis)

    return out

def to_spectrum(data, keep_dc=False):
    """
    Convert the time-series data into Fourier spectrum.
    """
    size, dof = data.shape
    spec = np.zeros((int(size/2), dof))
    dt = 1.0 # Time duration in seconds

    for d in range(dof):
        spec[:, d], _ = fourier_spectrum(data[:, d], dt, spectrum_type='power')

        # Filter the DC component
        if not keep_dc:
             spec[0, d] = 0.0

    return spec

def to_spectrum_old(data, keep_dc=False):

    num_dims = 3
    num_samples, num_features = data.shape
    size = int(num_features / num_dims)
    spec = np.zeros((num_samples, num_dims, int(size/2)))
    out = [[] for i in range(num_samples)]
    dt = 1.0 # Time duration in seconds

    for s in range(num_samples):
        for d in range(num_dims):
            ia = d*size
            iz = ia+size
            spec[s, d, :], _ = fourier_spectrum(data[s, ia:iz], dt, spectrum_type='power')

            # Filter the DC component
            if not keep_dc:
                spec[s, d, 0] = 0.0

        out[s] = np.hstack(spec[s])

    return np.array(out)

def normalize_ts(data, to='max'):
    """
    Normalize time-series data with shape of (num_tdata, DOF).
    """

    num_tdata, dof = data.shape
    arr_abs = np.linalg.norm(data, axis=0)
    max_abs = np.max(arr_abs)
    out = data / max_abs

    return out

def normalize(data, to='max'):

    num_samples, num_features = data.shape

    for i in range(num_samples):
        data[i,:] = data[i,:] / max(data[i,:])

    return data

