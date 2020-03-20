import numpy as np
import scipy.optimize
from dnn_play.activations import sigmoid, sigmoid_deriv
from dnn_play.utils.np_utils import to_binary_class_matrix, flatten_struct, pack_struct
from dnn_play.utils.gradient_utils import eval_numerical_gradient, rel_norm_diff


def ac_func(x):
    return sigmoid(x)

def ac_func_deriv(x):
    return sigmoid_deriv(x)

class SparseAutoencoder(object):
    """
    Sparse autoencoder.
    """

    def __init__(self, layer_units, weights=None):
        self.weights = weights
        self.layer_units = layer_units

    def init_weights(self):
        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        # Note layer_units[2] = layer_units[0]
        layer_units = self.layer_units
        n_layers = len(layer_units)
        assert n_layers == 3

        # Initialize parameters randomly based on layer sizes
        r  = np.sqrt(6) / np.sqrt(layer_units[1] + layer_units[0])

        # We'll choose weights uniformly from the interval [-r, r)
        weights = [{} for i in range(n_layers - 1)]
        weights[0]['W'] = np.random.random((layer_units[0], layer_units[1])) * 2.0 * r - r
        weights[1]['W'] = np.random.random((layer_units[1], layer_units[2])) * 2.0 * r - r
        weights[0]['b'] = np.zeros(layer_units[1])
        weights[1]['b'] = np.zeros(layer_units[2])

        self.weights = weights

        return self.weights


    def fit(self, X,
            reg=3e-3, beta=3, sparsity_param=1e-1,
            learning_rate=1e-2,
            optimizer='L-BFGS-B', max_iters=100,
            verbose=False):

        best_loss = 1e12
        best_weights = {}

        if self.weights is None:
            # lazily initialize weights
            self.weights = self.init_weights()

        # Solve with L-BFGS-B
        options = {'maxiter': max_iters, 'disp': verbose}

        def J(theta):
            weights = pack_struct(theta, self.layer_units)
            loss, grad = sparse_autoencoder_loss(weights, X, reg, beta=beta, sparsity_param=sparsity_param)
            grad = flatten_struct(grad)

            return loss, grad


        # Callback to get accuracies based on training set
        iter_feval = 0
        loss_history = []
        def progress(x):

             nonlocal iter_feval, best_weights, best_loss
             iter_feval += 1

             # Loss history
             weights = pack_struct(x, self.layer_units)
             loss, grad = sparse_autoencoder_loss(weights, X, reg, beta=beta, sparsity_param=sparsity_param)
             loss_history.append(loss)

             # Keep track of the best weights based on loss
             if loss < best_loss:
                 best_loss = loss
                 n_weights = len(weights)
                 best_weights = [{} for i in range(n_weights)]
                 for i in range(n_weights):
                     for p in weights[i]:
                         best_weights[i][p] = weights[i][p].copy()
             n_iters_verbose = max_iters / 20
             if iter_feval % n_iters_verbose == 0:
                 print("iter: {:4d}, loss: {:8f}".format(iter_feval, loss))

        # Minimize the loss function
        init_theta = flatten_struct(self.weights)
        results = scipy.optimize.minimize(J, init_theta, method=optimizer, jac=True, callback=progress, options=options)

        # Save weights
        self.weights = best_weights

        return self.weights, loss_history

    def predict(self, X):
        """
        Predict the outputs.
        """
        weights = self.weights

        # Weighting parameters
        W0 = weights[0]['W']
        b0 = weights[0]['b']
        W1 = weights[1]['W']
        b1 = weights[1]['b']

        # Number of samples
        m = X.shape[0]

        # Forward pass
        a0 = X
        z1 = np.dot(a0, W0) + b0
        a1 = ac_func(z1)
        z2 = np.dot(a1, W1) + b1
        a2 = ac_func(z2)

        return a2

    def forward_pass(self, X):
        """
        Perform forward pass and return activations of layer 1.
        """
        weights = self.weights

        # Weighting parameters
        W0 = weights[0]['W']
        b0 = weights[0]['b']

        # Number of samples
        m = X.shape[0]

        # Forward pass
        a0 = X                 # Input activation
        z1 = np.dot(a0, W0) + b0
        a1 = ac_func(z1)

        return a1

    def flatten_struct(self, data):
        return flatten_struct(data)

    def pack_struct(self, data):
        return pack_struct(data, self.layer_units)

    def get_weights(self):
        return self.weights

def sparse_autoencoder_loss(weights, X, reg, beta=3, sparsity_param=0.1):
    """
    Compute loss and gradients of the sparse autorncoder.
    """

    Y = X

    W0 = weights[0]['W']
    b0 = weights[0]['b']
    W1 = weights[1]['W']
    b1 = weights[1]['b']

    # Number of samples
    m = X.shape[0]

    # Forward pass
    a0 = X
    z1 = np.dot(a0, W0) + b0
    a1 = ac_func(z1)
    z2 = np.dot(a1, W1) + b1
    a2 = ac_func(z2)

    # Compute rho_hat used in sparsity penalty
    rho = sparsity_param
    rho_hat = np.mean(a1, axis=0)
    sparsity_delta = -rho/rho_hat + (1.0-rho)/(1-rho_hat)

    # Loss function
    mean_squared_error = 1.0/(2.0*m) * np.sum((a2 - Y)**2)
    weight_decay = 0.5*reg*(np.sum(W0*W0) + np.sum(W1*W1))
    sparsity_term = beta*np.sum(KL_divergence(rho, rho_hat))

    loss = mean_squared_error + weight_decay + sparsity_term

    # Backpropagation
    delta2 = (a2 - Y) * ac_func_deriv(z2)
    delta1 = (np.dot(delta2, W1.T) + beta*sparsity_delta) * ac_func_deriv(z1)

    # Gradients
    grad = [{} for i in range(len(weights))]
    grad[0]['W'] = np.dot(a0.T, delta1) / m + reg*W0
    grad[0]['b'] = np.mean(delta1, axis=0)
    grad[1]['W'] = np.dot(a1.T, delta2) / m + reg*W1
    grad[1]['b'] = np.mean(delta2, axis=0)

    return loss, grad

def KL_divergence(p, q):
    """
    Kullback-Leiber divergence.
    """

    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def rel_err_gradients():
    """
    Return the relative error between analytic and nemerical gradients.
    """
    # Number of layer units
    n_samples = 100
    input_size  = 4 * 4
    hidden_size = 4
    output_size = input_size
    layer_units = (input_size, hidden_size, output_size)

    X_train = np.random.randn(n_samples, input_size)
    reg = 1e-4
    beta = 3               # weight of sparsity penalty term
    sparsity_param = 1e-1  # desired average activation of the hidden units

    # Define the classifier
    sae = SparseAutoencoder(layer_units)

    # Initialize weights
    weights = sae.init_weights()

    # Analytic gradients of the cost function
    cost, grad = sparse_autoencoder_loss(weights, X_train, reg, beta=beta, sparsity_param=sparsity_param)
    grad = sae.flatten_struct(grad) # Flattened gradients

    def J(theta):
        # Structured weights
        weights = sae.pack_struct(theta)
        return sparse_autoencoder_loss(weights, X_train, reg, beta=beta, sparsity_param=sparsity_param)[0]

    theta = sae.flatten_struct(weights)
    numerical_grad = eval_numerical_gradient(J, theta)

    # Compare numerically computed gradients with those computed analytically
    rel_err = rel_norm_diff(numerical_grad, grad)

    return rel_err
