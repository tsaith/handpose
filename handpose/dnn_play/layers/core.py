import numpy as np

from .. import initializations
from .. import activations

class Layer(object):
    """
    Base Layer.
    """
    def __init__(self, **kwargs):
        self.previous = None
        self.catche = None

    def build(self):
        pass

    def set_previous(self, layer):
        self.previous = layer

    def get_output(self):
        pass

    def backpropagate(self):
        pass


class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(self, output_dim, input_dim=None, init='glorot_uniform',
                 activation='relu',
                 init_weights=None,
                 W_regularizer=None):

        """
        weights: initail weights.
        """

        self.output_dim = output_dim
        self.input_dim = input_dim

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.weights = weights

        self.W_regularizer = regularizers.get(W_regularizer)

    def build(self):

        # Initialize weights
        if not self.weights:
            self.W = self.init((self.input_dim, self.output_dim))
            self.b = np.zeros(self.output_dim)

            self.weights = [self.W, self.b]


    def get_weights(self):
        return self.weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(K.dot(X, self.W) + self.b)
        return output

    def init_weights(self, eps=1e-4):
        """
        Initialize weights.

        layer_units: tuple stores the size of each layer.
        weights: structured weights.
        """

        layer_units = self.layer_units

        n_layers = len(layer_units)

        weights = [{} for i in range(n_layers - 1)]
        for i in range(n_layers - 1):
            weights[i]['W'] = eps * np.random.randn(layer_units[i], layer_units[i+1])
            weights[i]['b'] = np.zeros(layer_units[i+1])

        self.weights = weights

        return self.weights

    def fit(self, X, y, X_val, y_val,
            reg=0.0,
            learning_rate=1e-2,
            optimizer='L-BFGS-B', max_iters=100,
            sample_batches=True,
            n_epochs=30, batch_size=128,
            verbose=False):

        epoch = 0
        best_val_acc = 0.0
        best_weights = {}

        if self.weights is None:
            # lazily initialize weights
            self.weights = self.init_weights()

        # Solve with L-BFGS-B
        options = {'maxiter': max_iters, 'disp': verbose}

        def J(theta):

            weights = pack_struct(theta, self.layer_units)
            loss, grad = neural_net_loss(weights, X, y, reg)
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
             loss, grad = neural_net_loss(weights, X, y, reg)
             loss_history.append(loss)

             # Training accurary
             y_pred_train = neural_net_predict(weights, X)
             train_acc = np.mean(y_pred_train == y)
             train_acc_history.append(train_acc)

             # Validation accuracy
             y_pred_val= neural_net_predict(weights, X_val)
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
        pred = neural_net_predict(self.weights, X)

        return pred

    def flatten_struct(self, data):
        return flatten_struct(data)

    def pack_struct(self, data):
        return pack_struct(data, self.layer_units)

def neural_net_loss(weights, X, y, reg):
    """
    Compute loss and gradients of the neutral network.
    """

    Y = to_binary_class_matrix(y)
    L = len(weights) # The index of the output layer
    z = []
    a = []

    # Number of samples
    m = X.shape[0]

    # Forward pass
    z.append(0)                 # Dummy element
    a.append(X)                 # Input activation

    for i in range(0, L):
        W = weights[i]['W']
        b = weights[i]['b']
        z.append(np.dot(a[-1], W) + b)
        a.append(ac_func(z[-1]))

    # loss function
    sum_weight_square = 0.0 # Sum of weight square
    for i in range(L):
        W = weights[i]['W']
        sum_weight_square += np.sum(W*W)

    loss = 1.0/(2.0*m) * np.sum((a[-1] - Y)**2) + 0.5*reg*sum_weight_square

    # Backpropagation
    delta = [(a[-1] - Y) * ac_func_deriv(z[-1])]
    for i in reversed(range(L)): # Note that delta[0] will not be used
        W = weights[i]['W']
        d = np.dot(delta[0], W.T) * ac_func_deriv(z[i])
        delta.insert(0, d) # Insert element at beginning

    # Gradients
    grad = [{} for i in range(L)]
    for i in range(L):
        W = weights[i]['W']
        grad[i]['W'] = np.dot(a[i].T, delta[i+1]) / m + reg*W
        grad[i]['b'] = np.mean(delta[i+1], axis=0)

    return loss, grad

def neural_net_predict(weights, X):
    """
    X: the N x M input matrix, where each column data[:, i] corresponds to
          a single test set

    pred: the predicted results.
    """

    L = len(weights) # The index of the output layer
    z = []
    a = []

    # Number of samples
    m = X.shape[0]

    # Forward pass
    z.append(0)                 # Dummy element
    a.append(X)                 # Input activation

    for i in range(0, L):
        W = weights[i]['W']
        b = weights[i]['b']
        z.append(np.dot(a[-1], W) + b)
        a.append(ac_func(z[-1]))

    # Predictions
    pred = np.argmax(a[-1], axis=1)

    return pred

def rel_err_gradients():
    """
    Return the relative error between analytic and nemerical gradients.
    """

    # Number of layer units
    n_samples = 100
    input_size  = 4 * 4
    hidden_size = 4
    n_classes = 10
    layer_units = (input_size, hidden_size, n_classes)

    X_train = np.random.randn(n_samples, input_size)
    y_train = np.random.randint(n_classes, size=n_samples)
    reg = 1e-4

    # Define the classifier
    clf = NeuralNet(layer_units)

    # Initialize weights
    weights = clf.init_weights()

    # Analytic gradients of the cost function
    cost, grad = neural_net_loss(weights, X_train, y_train, reg)
    grad = clf.flatten_struct(grad) # Flattened gradients

    def J(theta):
        # Structured weights
        weights = clf.pack_struct(theta)
        return neural_net_loss(weights, X_train, y_train, reg)[0]

    theta = clf.flatten_struct(weights)
    numerical_grad = eval_numerical_gradient(J, theta)

    # Compare numerically computed gradients with those computed analytically
    rel_err = rel_norm_diff(numerical_grad, grad)

    return rel_err

class Dense(Layer):
    """
    Fully connected layer.
    """

    def __init__(self, output_dim, input_dim=None, init='glorot_uniform', activation='ReLU', weights=None, **kwargs):

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.weights = weights

    def build(self):
        input_dim = self.input_dim

        self.W = self.init((input_dim, self.output_dim))
        self.b = np.zeros(self.output_dim)

        self.weights = [self.W, self.b]

    def get_weights(self):
        return self.weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        output = self.activation(K.dot(X, self.W) + self.b)
        return output
