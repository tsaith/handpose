from ..layers.core import Layer


class Sequential(object):
    '''
    The Sequential container is a linear stack of layers.
    '''
    def __init__(self):
        self.layers = []
        self.layer_cache = {}

        print("ttt")

    def add(self, layer):
        self.layers
        self.layers.append(layer)
        if len(self.layers) > 1:
            self.layers[-1].set_previous(self.layers[-2])
