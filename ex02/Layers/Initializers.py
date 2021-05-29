import numpy as np

class Constant:
    def __init__(self, weight_constant=0.1):
        self.weight_constant = weight_constant

    def initialize(self, weights_shape, fan_in, fan_out):
        #initialize weight matrix with a constant value
        final_weights = np.full(weights_shape,self.weight_constant)
        return final_weights

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        #initialize weights with uniformly distributed values
        #create matrix
        final_weights=np.zeros(weights_shape)
        #fill values
        final_weights=np.random.uniform(0,1,(weights_shape))

        return final_weights

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        #Xavier initialization: scale variance using fan_int & fan_out
        #calculate variance
        sigma = np.sqrt(2) / np.sqrt(fan_out + fan_in)
        #create matrix
        final_weights=np.zeros(weights_shape)
        #fill final_weight with normally distributed values
        final_weights=np.random.normal(0,sigma,(weights_shape))
        return final_weights

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        #He initialization
        sigma = np.sqrt(2 / fan_in)
        final_weights=np.zeros(weights_shape)
        final_weights=np.random.normal(0,sigma,(weights_shape))

        return final_weights
