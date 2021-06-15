import numpy as np
from Layers import Helpers
from Layers import Base
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        self.num_of_channels = channels
        self.trainable = True
        self.testing_phase = False
        self.iteration_count = 0

        self.weights = np.ones(self.num_of_channels)
        self.bias = np.zeros(self.num_of_channels)
        self.reg = False

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

        """On Batchnormalization, we do every operation in 2D Vector format.
        Therefore if the input is in 4D(Convolutional format), we convert it in
        2D vector format. If the input is already 2D, we keep it as it is."""
        if (len(input_tensor.shape) == 2):
            new_input = input_tensor
        else:
            new_input = self.reformat(input_tensor)

        if not self.testing_phase:
            """1. We calculate our mean and variance over batch axis."""
            self.mean = np.mean(new_input, axis=0)
            self.variance = np.var(new_input, axis=0)

            """2. We perform the ((X - mean)/np.sqrt(variance + e)) operation"""
            self.input_normalized = (new_input - self.mean)/np.sqrt(self.variance + (np.finfo(float).eps))

            """3. After getting the X-Tilda we perform ((weights * X_Tilda) + Bias)"""
            output = (self.weights * self.input_normalized) + self.bias

            """As mentioned before, we converted our format from 4D to 2D, if the
            shape of the input was 4D. Hence, we have to return the output in 4D
            format if the input was 4D. If the input is 2D, we keep the shape."""
            if (len(input_tensor.shape) > 2):
                output = self.reformat(output)

            """In the Batchnormalization, we also have to calculate moving mean and
            variance, in order to use them in the testing phase. It helps us to get
            better accuracy and also takes less time during testing."""

            """In the first iteration, we don't have any moving mean and variance. So
            we set the current mean and variance to the moving mean and variance."""
            if (self.iteration_count == 0):
                self.moving_mean = self.mean
                self.moving_var = self.variance

            self.moving_mean = (momentum * self.moving_mean) + ((1 - momentum) * self.mean)
            self.moving_var = (momentum * self.moving_var) + ((1 - momentum) * self.variance)
        else:

            """In the testing phase, want to calculate normalized input, but we use the
            moving mean and variance which we calculated in the training phase."""
            input_normalized = (new_input - self.moving_mean) / np.sqrt(self.moving_var + (np.finfo(float).eps))
            output = (self.weights * input_normalized) + self.bias

            """Again, before returning the output, we reformat it if the input came in
            4D format."""
            if (len(input_tensor.shape) > 2):
                output = self.reformat(output)

        self.iteration_count += 1
        return output

    def backward(self, error_tensor):
        """Repeating the procedure, in the backward method, we also convert all our inputs to 2D """
        if (len(self.input_tensor.shape) == 2):
            self.error_tensor = error_tensor
            t_input = self.input_tensor
        else:
            self.error_tensor = self.reformat(error_tensor)
            t_input = self.reformat(self.input_tensor)

        """We calculate the gradient w.r.t weights via getting the sum over the batch axis of (E*X_Tilda)
        For gradient w.r.t bias we just take the sum for the error tensor over the batch axis."""

        self.gradient_w = np.sum(self.error_tensor * self.input_normalized, axis=0)
        self.gradient_b = np.sum(self.error_tensor, axis=0)

        if (self._optimizer_b != None):
            self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_b)
        if (self._optimizer_w != None):
            self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_w)
            
        """For gradient w.r.t input, we pass the Helper function all the necessary input for calculating
        the output. As the Helper function can only work with vectors not convolutional, it will also
        return the output in the vector format. So, one last thing we have to do before returning the
        output, is to check the shape of  error or input tensor. If the shape is 4D, we have to reformat
        our vector output into convolutional format."""
        
        output = Helpers.compute_bn_gradients(self.error_tensor, t_input, self.weights, self.mean, self.variance)
        if (len(error_tensor.shape) > 2):
            output = self.reformat(output)
        return output

    def reformat(self, tensor):
        """The purpose of this function is to, reformat the tensor. If the tensor is in Convolutional format,
        it will reformat it into Vector format, and vise-versa."""
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

