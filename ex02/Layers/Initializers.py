import numpy as np

class Constant:
    def __init__(self, weight_constant=0.1):
        self.weight_constant = weight_constant

    def initialize(self, weights_shape, fan_in, fan_out):
        self.fan_in=fan_in
        self.fan_out=fan_out
        # if (len(weights_shape) == 2):
        #     no_rows, no_columns = weights_shape
        #     weights = np.full((no_rows*no_columns),self.weight_constant)
        #     final_weights = np.reshape(weights, (no_rows, no_columns))
        #
        # else:
        #     no_rows = weights_shape[0]
        #     weights = np.array(self.weight_constant,no_rows)
        #     final_weights = weights
        pass

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        if(len(weights_shape)==2):
            no_rows, no_columns = weights_shape
            weights = np.random.uniform(0, 1, (no_rows * no_columns))
            final_weights = np.reshape(weights, (no_rows, no_columns))

        else:
            no_rows = weights_shape[0]
            weights = np.random.uniform(0,1,no_rows)
            final_weights=weights

        self.fan_int = fan_in
        self.fan_out = fan_out
        return final_weights

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_out + fan_in)

        if (len(weights_shape)==2):
            no_rows, no_columns = weights_shape
            weights = np.random.normal(0, sigma, (no_rows * no_columns))
            final_weights = np.reshape(weights, (no_rows, no_columns))

        else:
            no_rows = weights_shape[0]
            weights = np.random.normal(0, sigma, no_rows)
            final_weights = weights
        self.fan_int = fan_in
        self.fan_out = fan_out
        return final_weights

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_in)

        if(len(weights_shape)==2):
            no_rows, no_columns = weights_shape
            weights = np.random.normal(0, sigma, (no_rows * no_columns))
            final_weights = np.reshape(weights, (no_rows, no_columns))

        else:
            no_rows = weights_shape[0]
            weights = np.random.normal(0, sigma, no_rows)
            final_weights = weights
        self.fan_int = fan_in
        self.fan_out = fan_out
        return final_weights
