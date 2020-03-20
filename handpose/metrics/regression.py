import numpy as np
from sklearn.metrics import mean_squared_error


def average_accuracy(y_true, y_pred, full_range=np.pi):
    """
    Average accuracy for regression tasks.

    Parameters:

    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns:

    accuracy: float
        The average accuracy.

    """  
    
    mse = mean_squared_error(y_true, y_pred)
    accuracy = 1.0 - mse/full_range

    return accuracy


