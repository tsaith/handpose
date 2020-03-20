import numpy as np

def check_numerical_gradient():
    """
    This code can be used to check your numerical gradient implementation
    in computeNumericalGradient.m
    It analytically evaluates the gradient of a very simple function called
    simpleQuadraticFunction (see below) and compares the result with your numerical
    solution. Your numerical gradient implementation is incorrect if
    your numerical solution deviates too much from the analytical solution.
    """

    # Evaluate the function and gradient at x = [4, 10]
    x = np.array([4, 10], dtype=np.float64)
    value, grad = simple_quadratic_function(x)

    # Use your code to numerically compute the gradient of simple_quadratic_function at x.
    func = lambda x : simple_quadratic_function(x)[0]
    numgrad = compute_numerical_gradient(func, x)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    n_grad = grad.size
    for i in range(n_grad):
        print("{0:20.12f} {1:20.12f}".format(numgrad[i], grad[i]))
    print('The above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be 2.1452e-12
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("Norm of difference = ", diff)
    print('Norm of the difference between numerical and analytical gradient (should be < 1e-9)\n\n')


def simple_quadratic_function(x):
    """
    This function accepts a vector as input.
    Its outputs are:
    value: h(x0, x1) = x0^2 + 3*x0*x1
    grad: A vector that gives the partial derivatives of h with respect to x0 and x1
    """

    value = x[0]*x[0] + 3*x[0]*x[1]

    grad = np.zeros(2)
    grad[0]  = 2*x[0] + 3*x[1]
    grad[1]  = 3*x[0]

    return value, grad

def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
  """
  a naive implementation of numerical gradient of f at x
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  #fx = f(x) # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    if verbose:
      print(ix, grad[ix])
    it.iternext() # step to next dimension

  return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
  """
  Evaluate a numeric gradient for a function that accepts a numpy
  array and returns a numpy array.

  df: upstream derivative
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index

    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval

    grad[ix] = np.sum((pos - neg) * df) / (2 * h)

    it.iternext()

  return grad


def rel_norm_diff(a, b):
    # Relative norm of difference. (L2 norm)

    return np.linalg.norm(a - b) / np.linalg.norm(a + b)
