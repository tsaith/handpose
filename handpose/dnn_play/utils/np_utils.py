import numpy as np
import scipy

def to_binary_class_matrix(y):
    """
    Convert class vector (integers from 0 to K)
    to binary class matrix.
    """
    m = y.size # Number of samples
    n_classes = np.max(y) + 1

    Y = scipy.sparse.csr_matrix((np.ones(m), (np.arange(m), y)))
    Y = np.array(Y.todense())

    return Y

def flatten_struct(data):
    """
    Flatten the data structure.
    """

    n_data = len(data)

    # Data vector
    data_vec = np.concatenate((data[0]['W'].flatten(), data[0]['b'].flatten()))
    out = data_vec

    for i in range(1, n_data):

        data_vec = np.concatenate((data[i]['W'].flatten(), data[i]['b'].flatten()))
        out = np.concatenate((out, data_vec))

    return out

def pack_struct(data, layer_units):
    """
    Pack the flattened data with structure.
    """

    n_layers = len(layer_units)
    struct = [{} for i in range(n_layers - 1)]

    iz = 0
    for i in range(n_layers - 1):
        ia = iz; iz = ia + layer_units[i]*layer_units[i+1]
        struct[i]['W'] = data[ia:iz].reshape((layer_units[i], layer_units[i+1]))
        ia = iz; iz = ia + layer_units[i+1]
        struct[i]['b'] = data[ia:iz]

    return struct

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
