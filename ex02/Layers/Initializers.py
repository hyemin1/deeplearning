import numpy as np

class Constant:
    def __init__(self, weight_constant=0.1):
        self.weight_constant = weight_constant

    def initialize(self, weights_shape, fan_in, fan_out):

        # if (len(weights_shape) == 2):
        #     no_rows, no_columns = weights_shape
        #     weights = np.full((no_rows*no_columns),self.weight_constant)
        #     final_weights = np.reshape(weights, (no_rows, no_columns))
        #
        # else:
        #     no_rows = weights_shape[0]
        #     weights = np.array(self.weight_constant,no_rows)
        #     final_weights = weights
        final_weights = np.full(weights_shape,self.weight_constant)
        return final_weights

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        final_weights=None
        # if(len(weights_shape)==2):
        #     no_rows, no_columns = weights_shape
        #     weights = np.random.uniform(0, 1, (no_rows * no_columns))
        #     final_weights = np.reshape(weights, (no_rows, no_columns))
        #
        # else:
        #     no_rows = weights_shape[0]
        #     weights = np.random.uniform(0,1,no_rows)
        #     final_weights=weights
        if (len(weights_shape)==4):
            final_weights=np.zeros((weights_shape[0],weights_shape[1],weights_shape[2],weights_shape[3]))
            for ker in range(weights_shape[0]):
                for channel in range(weights_shape[1]):
                    final_weights[ker,channel]=np.random.uniform(0, 1, (weights_shape[2] , weights_shape[3]))
        elif(len(weights_shape==3)):
            final_weights = np.zeros((weights_shape[0], weights_shape[1], weights_shape[2]))
            for ker in range(weights_shape[0]):
                for channel in range(weights_shape[1]):
                    final_weights[ker, channel] = np.random.uniform(0, 1, (weights_shape[2]))
        else:
            for ker in range(weights_shape[0]):
                final_weights[ker] = np.random.normal(0, 1,1)

        return final_weights

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2) / np.sqrt(fan_out + fan_in)
        final_weights=None
        # if (len(weights_shape)==2):
        #     no_rows, no_columns = weights_shape
        #     weights = np.random.normal(0, sigma, (no_rows * no_columns))
        #     final_weights = np.reshape(weights, (no_rows, no_columns))
        #
        # else:
        #     no_rows = weights_shape[0]
        #     weights = np.random.normal(0, sigma, no_rows)
        #     final_weights = weights
        if (len(weights_shape)==4):
            final_weights=np.zeros((weights_shape[0],weights_shape[1],weights_shape[2],weights_shape[3]))
            for ker in range(weights_shape[0]):
                for channel in range(weights_shape[1]):
                    final_weights[ker,channel]=np.random.normal(0, sigma, (weights_shape[2] , weights_shape[3]))
        elif(len(weights_shape==3)):
            final_weights = np.zeros((weights_shape[0], weights_shape[1], weights_shape[2]))
            for ker in range(weights_shape[0]):
                for channel in range(weights_shape[1]):
                    final_weights[ker, channel] = np.random.normal(0, sigma, (weights_shape[2]))
        else:
            for ker in range(weights_shape[0]):
                final_weights[ker] = np.random.normal(0, sigma,1)
        return final_weights

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        final_weights=None
        # if(len(weights_shape)==2):
        #     no_rows, no_columns = weights_shape
        #     weights = np.random.normal(0, sigma, (no_rows * no_columns))
        #     final_weights = np.reshape(weights, (no_rows, no_columns))
        #
        # else:
        #     no_rows = weights_shape[0]
        #     weights = np.random.normal(0, sigma, no_rows)
        #     final_weights = weights
        if (len(weights_shape)==4):
            final_weights=np.zeros((weights_shape[0],weights_shape[1],weights_shape[2],weights_shape[3]))
            for ker in range(weights_shape[0]):
                for channel in range(weights_shape[1]):
                    final_weights[ker,channel]=np.random.normal(0, sigma, (weights_shape[2] , weights_shape[3]))
        elif(len(weights_shape==3)):
            final_weights = np.zeros((weights_shape[0], weights_shape[1], weights_shape[2]))
            for ker in range(weights_shape[0]):
                for channel in range(weights_shape[1]):
                    final_weights[ker, channel] = np.random.normal(0, sigma, (weights_shape[2]))
        else:
            for ker in range(weights_shape[0]):
                final_weights[ker] = np.random.normal(0, sigma,1)

        return final_weights
