import numpy as np

def get_velocity(accel, dt, v0):
    """
    Estimate the velocity.
    
    Parameters
    ----------
    accel: array
        Acceleration.
    dt: float
        Time duration.    
    v0: array
        Initial velocity.
        
    Returns
    -------
    v: array
        Velocity.
    """
    dv = accel * dt    
    v = v0 + dv
    
    return v
    
def get_dx(v, dt):
    """
    Estimate the spatial displacement.
    
    Parameters
    ----------
    v: array
        Velocity.
    dt: float
        Time duration.  
        
    Reurns
    ------
    dx: array
        Spatial displacement. 
    """
    dx = v * dt    
    
    return dx    
    
