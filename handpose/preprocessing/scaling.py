import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, \
                                  MinMaxScaler, RobustScaler

def scale(X, method='standard'):
    """
    Scale and center the data.
    
    Parameters
    ----------
    X: array-like
        The data to center and scale..
    
    method: str
        The method for scaling; 'standard' is the default. 
        'standard': Standardize features by removing the mean and 
                    scaling to unit variance. 
        'minmax': Transforms features by scaling each feature 
                  to a given range. Default range is (0, 1).
        'robust': Scale features using statistics that are robust to outliers.
    
    Returns
    -------
    scaled: array-like
        Data after scaling and centering.
        
    scaler: object
        Scaler used to scale and center data. 
    """
 
    # Define scalers
    standard_scaler = StandardScaler()     
    minmax_scaler = MinMaxScaler()     
    robust_scaler = RobustScaler()

    scaler = {
        'standard': standard_scaler,
        'minmax': minmax_scaler,
        'robust': robust_scaler
    }.get(method)
    
    scaled = scaler.fit_transform(X)
    
    return scaled, scaler

