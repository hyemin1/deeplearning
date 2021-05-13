import numpy as np

class Constant:
    def __init__(self, weight_constant=0.1):
        self.weight_constant = weight_constant

    def initialize(self, weights_shape, fan_in, fan_out):
        pass

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        no_rows, no_columns = weights_shape
        weights = np.random.uniform(0, 1, (no_rows * no_columns))
        final_weights = np.reshape(weights, (no_rows, no_columns))

        return final_weights

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_out + fan_in)

        no_rows, no_columns = weights_shape
        weights = np.random.normal(0, sigma, (no_rows * no_columns))
        final_weights = np.reshape(weights, (no_rows, no_columns))

        return final_weights

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in)

        no_rows, no_columns = weights_shape
        weights = np.random.normal(0, sigma, (no_rows * no_columns))
        final_weights = np.reshape(weights, (no_rows, no_columns))

        return final_weights