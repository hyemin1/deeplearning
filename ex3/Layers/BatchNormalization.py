import numpy as np
import Helpers
from Layers import Base
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        self.num_of_channels = channels
        self.trainable = True
        self.testing_phase = False

        self.weights = np.ones(self.num_of_channels)
        self.bias = np.zeros(self.num_of_channels)

        self._optimizer_b = None
        self._optimizer_w = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights=np.ones((self.weights.shape))
        self.bias=np.zeros((self.bias.shape))

    @property
    def gradient_weights(self):
        return self.gradient_w

    @property
    def gradient_bias(self):
        return self.gradient_b

    @property
    def optimizer(self):
        return self._optimizer_w

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer_b = copy.deepcopy(opt)
        self._optimizer_w = copy.deepcopy(opt)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        momentum = 0.8

        if (len(input_tensor.shape) == 2):
            new_input = input_tensor
        else:
            new_input = self.reformat(input_tensor)

        if not self.testing_phase:
            self.mean = np.mean(new_input, axis=0)
            self.variance = np.var(new_input, axis=0)

            self.input_normalized = (new_input - self.mean)/np.sqrt(self.variance + (np.finfo(float).eps))
            output = (self.weights * self.input_normalized) + self.bias

            if (len(input_tensor.shape) > 2):
                output = self.reformat(output)

            self.moving_mean = self.mean
            self.moving_var = self.variance

            self.moving_mean = (momentum * self.moving_mean) + ((1 - momentum) * self.mean)
            self.moving_var = (momentum * self.moving_var) + ((1 - momentum) * self.variance)
        else:
            input_normalized = (new_input - self.moving_mean) / np.sqrt(self.moving_var + (np.finfo(float).eps))
            output = (self.weights * input_normalized) + self.bias

            if (len(input_tensor.shape) > 2):
                output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        if (len(self.input_tensor.shape) == 2):
            self.error_tensor = error_tensor
            t_input = self.input_tensor
        else:
            self.error_tensor = self.reformat(error_tensor)
            t_input = self.reformat(self.input_tensor)

        self.gradient_w = np.sum(self.error_tensor * self.input_normalized, axis=0)
        self.gradient_b = np.sum(self.error_tensor, axis=0)

        if (self._optimizer_b != None):
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_b)
        if (self._optimizer_w != None):
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_w)

        output = Helpers.compute_bn_gradients(self.error_tensor, t_input, self.weights, self.mean, self.variance)
        if (len(error_tensor.shape) > 2):
            output = self.reformat(output)
        return output

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
