import numpy as np

class Regularizer(object):
    def set_param(self, p):
        self.p = p

    def set_layer(self, layer):
        self.layer = layer

    def __call__(self, loss):
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__}


class WeightRegularizer(Regularizer):
    def __init__(self, l2=0.01):
        self.l2 = l2

    def set_param(self, p):
        self.p = p

    def __call__(self, loss):
        loss += np.sum(np.square(self.p)) * self.l2
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__,
                "l2": self.l2}

def l2(l=0.01):
    return WeightRegularizer(l2=l)



from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'regularizer')
