import numpy as np
import pandas as pd
import os
from ..utils import load_csv_file, list_files

def accel_abs(accel):
    
    ax = accel[:,0]
    ay = accel[:,1]
    az = accel[:,2]
    
    a_abs = np.sqrt(ax*ax + ay*ay + az*az)
    
    return a_abs
    
def find_seg_indexes(data, size=None, noise_ratio=5.0, back = 50):
    """
    Find the indexes of segmentations.
    """
    
    num_samples = len(data)
    
    num_noise = 200 # Nummbet of noise data
    
    noise_level = np.mean(data[:num_noise])
    
    threshold = noise_level * noise_ratio
    
    # Start and end index of the segmentations
    ia_indexes = []
    iz_indexes = []
    
    scan_start = num_noise
    scan_end = num_samples - int(0.5*size)
    i = scan_start
    while i < scan_end:
        
        i += 1
        if data[i] > threshold:
            ia_indexes.append(i-back)
            i += size
    
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
                     seg_size=None, noise_ratio=1.5, back=100, verbose=0):
    """
    File factory of vibrational data.
    """

    files = list_files(dir_path=dir_path, keyword=keyword)
    num_files = len(files)

    dir_out = "output"

    print("Start to generate files under the {} direcory.".format(dir_out))
    for f in files:

        # Load raw data
        fpath = os.path.join(dir_path, f)
        accel = load_csv_file(fpath)

        # Find the indexes of segmentations
        ax = accel[:,0]
        ay = accel[:,1]
        az = accel[:,2]
        a_abs = np.sqrt(ax*ax + ay*ay + az*az)
        ia_indexes, iz_indexes = find_seg_indexes(a_abs, size=seg_size, noise_ratio=noise_ratio, back=back)

        # Prepare the output data
        num_rows = len(ia_indexes)
        num_cols = 3*seg_size
        data = np.zeros((num_rows, num_cols))


        i = 0
        for ia, iz in zip (ia_indexes, iz_indexes):
            data[i, :] = np.hstack((accel[ia:iz,0], accel[ia:iz,1], accel[ia:iz,2]))
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

    print("There are {} files generated.".format(num_files))

    return 0
