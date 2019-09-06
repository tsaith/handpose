from keras.datasets import mnist
import numpy as np

def load_mnist(n_train=55000, n_val=5000, n_test=10000):
    """
    Load MNIST and return data for training, crosss-validation and testing.
    """

    # Load MNIST data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    mask = range(n_train, n_train + n_val)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(n_train)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(n_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Read image as data vector, shape (:, n_samples)
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val   = X_val.reshape((X_val.shape[0], -1))
    X_test  = X_test.reshape((X_test.shape[0], -1))

    # Change data type
    X_train = X_train.astype('float64')
    X_val   = X_val.astype('float64')
    X_test  = X_test.astype('float64')

    # Scale to [0, 1]
    X_train /= 255.0
    X_val   /= 255.0
    X_test  /= 255.0

    # Transpose
    #X_train = X_train.T
    #X_val   = X_val.T
    #X_test  = X_test.T

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
