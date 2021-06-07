import numpy as np
from Layers import Base

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        self.num_of_channels = channels
        self.trainable = True
        self.testing_phase = False

        self.weights = np.ones(self.num_of_channels)
        self.bias = np.zeros(self.num_of_channels)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights)
        self.bias = bias_initializer.initialize(self.bias)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        momentum = 0.8
        moving_mean = np.zeros((1, self.num_of_channels, 1, 1))
        moving_var = np.zeros((1, self.num_of_channels, 1, 1))

        if not self.testing_phase:
            if (len(input_tensor.shape) == 2):
                mean = np.mean(input_tensor, axis=0)
                variance = np.var(input_tensor, axis=0)

                moving_mean = np.zeros((1, self.num_of_channels))
                moving_var = np.zeros((1, self.num_of_channels))
            else:
                mean = np.mean(input_tensor, axis=(1, 2, 3), keepdims=True)
                variance = np.var(input_tensor, axis=(1, 2, 3), keepdims=True)

                self.weights = np.ones(input_tensor.shape)
                self.bias = np.zeros(input_tensor.shape)

            input_normalized = (input_tensor - mean) / np.sqrt(variance + (np.finfo(float).eps))
            moving_mean = momentum * moving_mean + ((1.0 - momentum) * mean)
            moving_var = momentum * moving_mean + ((1.0 - momentum) * mean)

            output = (self.weights * input_normalized) + self.bias
        else:
            if (len(input_tensor.shape) > 2):
                self.weights = np.ones(input_tensor.shape)
                self.bias = np.zeros(input_tensor.shape)

            input_normalized = (input_tensor - moving_mean) / np.sqrt(moving_var + (np.finfo(float).eps))
            output = (self.weights * input_normalized) + self.bias
        return output

    def backward(self, error_tensor):
        pass

    def reformat(self, tensor):
        if (len(tensor.shape) > 2):
            batch, input_channel, height, width = tensor.shape
            output = np.reshape(tensor, ((batch, input_channel, height * width)))
            output = output.transpose(0, 2, 1)
            output = np.reshape(output, ((batch * height * width), input_channel))
        else:
            batch, input_channel, height, width = self.input_tensor.shape
            output = np.reshape(tensor, (batch, height*width, input_channel))
            output = output.transpose(0, 2, 1)
            output = np.reshape(output, (batch, input_channel, height, width))

        return output

