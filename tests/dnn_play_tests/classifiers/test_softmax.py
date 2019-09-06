import pytest
import numpy as np
from numpy.testing import assert_allclose

from dnn_play.classifiers.softmax import Softmax, softmax_loss, rel_err_gradients

def test_gradients():
    """
    Test if the gradients is correct.
    """
    rel_err = rel_err_gradients()
    assert rel_err < 1e-8

class TestSoftmax:

    def test_predict(self):

        (X_train, y_train), (X_val, y_val, X_test, y_test) = _load_data()

        input_units  = X_train.shape[1]
        output_units = np.max(y_train) + 1
        layer_units = (input_units, output_units)

        reg = 1e-4

        clf = Softmax(layer_units)
        weights, loss_history, train_acc_history, val_acc_history = clf.fit(X_train, y_train, X_val, y_val,
                                                                            reg=reg, max_iters=200)

        pred = clf.predict(X_test)
        acc = np.mean(y_test == pred)

        assert acc > 0.8

def _load_data():

    n_train = 100
    n_val   = 100
    n_test  = 100

    input_units = 10
    output_units = 10
    layer_units = (input_units, output_units)

    X_train = np.zeros((n_train, input_units))
    y_train = np.random.randint(output_units, size=n_train)

    X_val   = np.zeros((n_val, input_units))
    y_val   = np.random.randint(output_units, size=n_val)

    X_test  = np.zeros((n_test, input_units))
    y_test  = np.random.randint(output_units, size=n_test)

    for i in range(n_train):
        X_train[i] = y_train[i]
    for i in range(n_val):
        X_val[i] = y_val[i]
    for i in range(n_test):
        X_test[i] = y_test[i]

    return (X_train, y_train), (X_val, y_val, X_test, y_test)
