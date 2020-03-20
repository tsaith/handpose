import numpy as np

def window_method(data_in, win_size=1):
    """
    The window method used in time series problem.

    Parameters

    data_in: input data array.
    win_size: window size which indicates how many 
              previous time steps will be considered.

    Return

    data: numpy-array, data applied window method.

    """ 


    rows_in = len(data_in)
    data = []
    for i in range(rows_in-win_size):
        tmp = data_in[i:i+win_size+1, :]
        data.append(np.concatenate(tmp))

    data =  np.array(data)    

    return data
