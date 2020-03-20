import numpy as np
import scipy.optimize
from dnn_play.utils.np_utils import to_binary_class_matrix, flatten_struct, pack_struct
from dnn_play.utils.gradient_utils import eval_numerical_gradient, rel_norm_diff

class Softmax(object):
    """
    Softmax classifer
    """

    def __init__(self, layer_units, weights=None):
        self.weights = weights
        self.layer_units = layer_units

    def init_weights(self, eps=1e-4):
        self.weights = init_weights(self.layer_units, eps=eps)
        return self.weights


    def fit(self, X, y, X_val, y_val,
              reg=0.0,
              optimizer='L-BFGS-B', max_iters=100,
              verbose=False):

        epoch = 0
        best_val_acc = 0.0
        best_weights = {}

        if self.weights is None:
            # lazily initialize weights
            self.weights = init_weights(self.layer_units)

        # Solve with L-BFGS-B
        options = {'maxiter': max_iters, 'disp': verbose}

        def J(theta):

            weights = pack_struct(theta, self.layer_units)
            loss, grad = softmax_loss(weights, X, y, reg)
            grad = flatten_struct(grad)

            return loss, grad

        # Callback to get accuracies based on training / validation sets
        iter_feval = 0
        loss_history = []
        train_acc_history = []
        val_acc_history   = []
        def progress(x):

             nonlocal iter_feval, best_weights, best_val_acc
             iter_feval += 1

             # Loss history
             weights = pack_struct(x, self.layer_units)
             loss, grad = softmax_loss(weights, X, y, reg)
             loss_history.append(loss)

             # Training accurary
             y_pred_train = softmax_predict(weights, X)
             train_acc = np.mean(y_pred_train == y)
             train_acc_history.append(train_acc)

             # Validation accuracy
             y_pred_val= softmax_predict(weights, X_val)
             val_acc = np.mean(y_pred_val == y_val)
             val_acc_history.append(val_acc)

             # Keep track of the best weights based on validation accuracy
             if val_acc > best_val_acc:
                 best_val_acc = val_acc
                 n_weights = len(weights)
                 best_weights = [{} for i in range(n_weights)]
                 for i in range(n_weights):
                     for p in weights[i]:
                         best_weights[i][p] = weights[i][p].copy()

             n_iters_verbose = max_iters / 20
             if iter_feval % n_iters_verbose == 0:
                 print("iter: {:4d}, loss: {:8f}, train_acc: {:4f}, val_acc: {:4f}".format(iter_feval, loss, train_acc, val_acc))

        # Minimize the loss function
        init_theta = flatten_struct(self.weights)
        results = scipy.optimize.minimize(J, init_theta, method=optimizer, jac=True, callback=progress, options=options)

        # Save weights
        self.weights = best_weights

        return self.weights, loss_history, train_acc_history, val_acc_history

    def predict(self, X):
        """
        X: the N x M input matrix, where each column data[:, i] corresponds to
              a single test set

        pred: the predicted results.
        """

        pred = softmax_predict(self.weights, X)

        return pred

    def flatten_struct(self, data):
        return flatten_struct(data)

    def pack_struct(self, data):
        return pack_struct(data, self.layer_units)

    def get_weights(self):
        return self.weights


def softmax_loss(weights, X, y, reg):
    """
    Compute the loss and derivative.

    theta: weight matrix
    X: the N x M input matrix, where each column data[:, i] corresponds to
          a single test set
    y: labels corresponding to the input data
    """

    # Small constant used to avoid numerical problem
    eps = 1e-10

    # Weighting parameters
    W0 = weights[0]['W']
    b0 = weights[0]['b']

    # Number of samples
    m = X.shape[0]

    # Forward pass
    a0 = X                 # Input activation
    z1 = np.dot(a0, W0) + b0

    z1_max = np.max(z1, axis=1, keepdims=True)
    z1 -= z1_max # Avoid numerical problem due to large values of exp(z1)
    proba = np.exp(z1) / np.sum(np.exp(z1), axis=1, keepdims=True) + eps # Add eps to avoid this value too close to zero

    # Target matrix of labels
    target = to_binary_class_matrix(y)

    # loss function
    loss = -1.0/m * np.sum(target * np.log(proba)) + 0.5*reg*np.sum(W0*W0)

    # Gradients
    delta1 = -1.0 * (target - proba)

    grad = [{}]
    grad[0]['W'] = np.dot(a0.T, delta1)/m + reg*W0
    grad[0]['b'] = np.mean(delta1, axis=0)

    return loss, grad

def softmax_predict(weights, X):
    """
    weights: weights trained using softmax_train
    X: the test matrix, where each column X[:, i] corresponds to
       a single test set

    pred: the prediction array.
    """

    # Small constant used to avoid numerical problem
    eps = 1e-10

    # Weighting parameters
    W0 = weights[0]['W']
    b0 = weights[0]['b']

    # Number of samples
    m = X.shape[1]

    # Forward pass
    a0 = X                 # Input activation
    z1 = np.dot(a0, W0) + b0

    # Propabilities
    z1_max = np.max(z1, axis=1, keepdims=True)
    z1 -= z1_max # Avoid numerical problem due to large values of exp(z1)
    proba = np.exp(z1) / np.sum(np.exp(z1), axis=1, keepdims=True) + eps # Add eps to avoid this value too close to zero

    # Predictions
    pred = np.argmax(proba, axis=1)

    return pred

def init_weights(layer_units, eps=1e-4):
    """
    Initialize weights.

    layer_units: tuple stores the size of each layer.
    weights: structured weights.
    """

    assert len(layer_units) == 2

    weights = [{}]
    weights[0]['W'] = eps * np.random.randn(layer_units[0], layer_units[1])
    weights[0]['b'] = np.zeros(layer_units[1])

    return weights


def rel_err_gradients():
    """
    Return the relative error between analytic gradients and nemerical ones.
    """

    # Number of layer units
    input_size  = 4 * 4
    hidden_size = 4
    n_classes = 10

    layer_units = (input_size, n_classes)

    X_train = np.random.randn(100, input_size)
    y_train = np.random.randint(n_classes, size=100)
    reg = 1e-4

    # Define the classifier
    clf = Softmax(layer_units)

    # Initialize weights
    weights = clf.init_weights()

    # Analytic gradients of the cost function
    cost, grad = softmax_loss(weights, X_train, y_train, reg)
    grad = clf.flatten_struct(grad) # Flattened gradients

    def J(theta):
        # Structured weights
        weights = clf.pack_struct(theta)
        return softmax_loss(weights, X_train, y_train, reg)[0]

    theta = clf.flatten_struct(weights)
    numerical_grad = eval_numerical_gradient(J, theta)

    # Compare numerically computed gradients with those computed analytically
    rel_err = rel_norm_diff(numerical_grad, grad)

    return rel_err
