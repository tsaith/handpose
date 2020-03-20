import numpy as np
import scipy

def sigmoid(x):
    """
    Return the sigmoid (aka logistic) function, 1 / (1 + exp(-x)). 
    """
    return scipy.special.expit(x)



