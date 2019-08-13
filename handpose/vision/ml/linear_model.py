import numpy as np
from scipy import optimize
from ..special import sigmoid

def logistic_cost(theta, X, y, C=1.0, tol=1e-4):
    """
    Retrun the cost function of logistic regression with binary classes.

    theta:
        Fitting parameters.
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
       n_features is the number of features.

    y : array-like, shape (n_samples,)
       Target vector relative to X.

    Returns
    -------
    J : float
        The cost function.

    """

    err_tol = 1e-20 # Error tolerance

    C_inv = 1.0 / C # Regulation coefficient
    n_samples = len(y)
    n_features = len(theta)

    m = n_samples
    n = n_features

    # Hypothesis function
    h = sigmoid(X.dot(theta))

    # Cost
    term_ori = -1.0/m * np.sum(y*np.log(h + err_tol) + (1.0-y)*np.log(1.0-h + err_tol)) # add a small value to avoid log(0) situation
    term_reg = 0.5*C_inv/m*(theta.dot(theta) - theta[0]*theta[0])
    J = term_ori + term_reg

    return J

def logistic_predict_proba(X, optimal_theta):
    """
    The probability estimates of logistic regression.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Observation vector, where n_samples is the number of samples and
       n_features is the number of features.

    optimal_theta: array, shape (n_features)
        optimal fitting parameters.
    """

    proba = sigmoid(X.dot(optimal_theta))

    return proba


class LogisticRegression:
    """
    Logistic regression.
    fit: fit the model.
    predict: make prediction.
    """

    def __init__(self, tol=1e-4, C=1.0, multi_class='ovr', disp_details=False):

        self._tol = tol
        self._C = C
        self._multi_class = multi_class
        self._disp_details = disp_details

        self._labels = None
        self._coef   = None
        self._err_tol = 1e-20 # Error tolence

    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        """

        self._labels = np.unique(y)
        n_samples, n_features  = X.shape
        n_classes = self._labels.size


        if n_classes == 2: # Binary problem
            n_classes = 1

        self._coef = np.zeros((n_classes, n_features))

        coef_init = np.zeros(n_features) # Initial coefficient
        for i in np.arange(n_classes):

            if n_classes == 1: # Binary case
                y_class = y
            else:
                y_class = y == self._labels[i]

            cost = lambda coef: logistic_cost(coef, X, y_class, C=self._C)
            results = optimize.minimize(cost, coef_init, method='BFGS', options={'disp': self._disp_details})
            self._coef[i, :] = results['x']

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

         X : {array-like, sparse matrix}, shape (n_samples, n_features)
             Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        labels : array, shape = [n_samples]
            Predicted class label per sample.
        """

        n_classes = self._coef.shape[0]

        if X.ndim == 1:
            n_samples = 1
        else:
            n_samples = X.shape[0]

        proba = self.predict_proba(X)

        labels = np.empty(n_samples, dtype=object)
        if n_classes == 1:
            for i in np.arange(n_samples):
                p = proba[i, 0]
                if p > 0.5:
                    labels[i] = True
                else:
                    labels[i] = False
        else:
            for i in np.arange(n_samples):
                class_index = np.argmax(proba[i, :])
                labels[i] = self._labels[class_index]

        return labels

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the label of classes.
        """

        n_classes = self._coef.shape[0]

        if X.ndim == 1:
            n_samples = 1
        else:
            n_samples = X.shape[0]

        proba = np.zeros((n_samples, n_classes))
        for i in np.arange(n_samples):
            for j in np.arange(n_classes):
                x = X[i, :]
                proba[i, j] = logistic_predict_proba(x, self._coef[j, :])


        return proba
