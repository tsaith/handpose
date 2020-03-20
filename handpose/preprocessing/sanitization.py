import numpy as np
import pandas as pd

def detect_na(data):
    """
    Detect if missing value exists. (NA or NaN)
    
    Parameters
    ----------
    x: ndarray or data frame
        Data to be detected.
    
    Returns
    -------
    has_na: bool
        Missing value exists or not.
    """
 
    df = pd.DataFrame(data)
    has_na = pd.isnull(df).values.any()

    return has_na
    
    
def fill_na(x, value=0.0, method=None):
    """
    
    Parameters
    ----------
    x: array-like
        Array to be processed.
    
    value : scalar
        Value to use to fill holes (e.g. 0).
    method : {None}, default None
        Method to use for filling missing values.
    
    Returns
    -------
    filled: ndarray
        Array with filled values.
    
    """
    
    df = pd.DataFrame(x)
    filled = df.fillna(value).values
    
    return filled
    
