import pytest
import numpy as np
from dnn_play.layers import affine_forward, affine_backward, relu_forward, relu_backward, softmax_loss, svm_loss
from dnn_play.utils.np_utils import rel_error
from dnn_play.utils.gradient_utils import eval_numerical_gradient, eval_numerical_gradient_array

def test_affine_forward():
    # Test the affine_forward function

    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                            [ 3.25553199,  3.5141327,   3.77273342]])

    assert rel_error(out, correct_out) < 1e-9


def test_affine_backward():
    # Test the affine_backward function

    x = np.random.randn(10, 6)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    dx_error = rel_error(dx_num, dx)
    dw_error = rel_error(dw_num, dw)
    db_error = rel_error(db_num, db)
    max_error = max(dx_error, dw_error, db_error)

    assert max_error < 1e-7

def test_relu_forward():
    # Test the relu_forward function.

    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    out, _ = relu_forward(x)
    correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                            [ 0.,          0.,          0.04545455,  0.13636364,],
                            [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])

    assert rel_error(out, correct_out) < 1e-7

def test_relu_backward():
    # Test the relu_backward function.

    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

    _, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    assert rel_error(dx, dx_num) < 1e-10

def test_softmax_loss():
    # Test softmax_loss function.

    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    loss, dx = softmax_loss(x, y)

    assert rel_error(dx, dx_num) < 1e-8

def test_svm_loss():
    # Test svm_loss function.

    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
    loss, dx = svm_loss(x, y)

    assert rel_error(dx, dx_num) < 1e-8
