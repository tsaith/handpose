import numpy as np

def get_distance(p0, p1):

    v = p1 - p0
    d = np.linalg.norm(v)

    return d
