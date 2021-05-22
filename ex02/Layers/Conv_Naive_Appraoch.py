from Layers import Base
import numpy as np
import math
from scipy import ndimage
from scipy import signal
from Layers import Initializers


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.weights = None
        self.bias = None
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape  # shape for convolution
        self.num_kernels = num_kernels  # total number of kernels
        # optimizer to update wdight
        self.opt_w = None
        # optimizer to update bias
        self.opt_b = None
        self.img_2d = False

        # number of input channels
        self.input_cha_num = self.convolution_shape[0]

        # in 1D: length of array, in 2D: height of matrix
        self.m = self.convolution_shape[1]

        # in case of 2D input
        if (len(convolution_shape) == 3):
            self.img_2d = True
            # width of matrix
            self.n = self.convolution_shape[2]
            # initialize weights using given kernel shape
            self.weights = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m * self.n))
            self.weights = np.reshape(self.weights, (self.num_kernels, self.input_cha_num, self.m, self.n))
            # initialize bias
            # self.bias = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m * self.n))
            # self.bias = np.reshape(self.bias, (self.num_kernels, self.input_cha_num, self.m, self.n))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
            # self.bias = np.reshape(self.bias,(self.num_kernels,self.input_cha_num))
        # in case of 1D input
        else:
            # initialize weights using given kernel shape
            self.weights = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m))
            self.weights = np.reshape(self.weights, (self.num_kernels, self.input_cha_num, self.m))
            # initialize bias
            # self.bias = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
            # self.bias = np.reshape(self.bias, (self.num_kernels, self.input_cha_num))

    @property
    def gradient_weights(self):
        return self.gradient_w

    @property
    def gradient_bias(self):
        return self.gradient_b

    @property
    def optimizer(self, opt):
        self.opt_w = np.copy.deepcopy(opt)
        self.opt_b = np.copy.deepcopy(opt)

    # should be corrected
    def initialize(self, weights_initializer, bias_initializer):

        if (self.img_2d == True):
            self.fan_in = self.input_cha_num * self.m * self.n
            self.fan_out = self.num_kernels * self.m * self.n

            weights_initializer.initialize((self.m, self.n), self.fan_in, self.fan_out)
            bias_initializer.initialize((1), self.fan_in, self.fan_out)
        else:
            self.fan_in = self.input_cha_num * self.m
            self.fan_out = self.num_kernels * self.m

            weights_initializer.initialize((self.m), self.fan_in, self.fan_out)
            bias_initializer.initialize((1), self.fan_in, self.fan_out)

    def forward(self, input_tensor):
        """
        implementation step
        0. set input layout
        1. calculate the output tensor shape
        2. do zero padding
        3. do convolution using scipy library,weight matrix,bias
        """

        """
        0. set input layout
        """
        self.input_tensor = input_tensor
        # number of batches
        b = input_tensor.shape[0]
        # set dimension(number of input channels per batch)
        self.c = input_tensor.shape[1]
        # spatial dimension
        # in 1D: length of array, in 2D: height of matrix
        y = input_tensor.shape[2]
        # in 2D
        if (self.img_2d == True):
            # width of matrix
            x = input_tensor.shape[3]

        """
        1. calculate the output tensor shape
        2. do zero padding
        """
        if (self.img_2d == True):
            pad_height = np.ceil((self.stride_shape[0] * y - self.stride_shape[0] + self.m - y))
            pad_height = int(pad_height)
            pad_width = np.ceil((self.stride_shape[1] * x - self.stride_shape[1] + self.n - x))
            pad_width = int(pad_width)

            # padded_input = np.pad(input_tensor,[(0,0),(pad_height,pad_height),(pad_width,pad_width),(0,0)],mode='constant',constant_values=0)

            out_height = np.ceil(float(y) / float(self.stride_shape[0]))
            out_width = np.ceil(float(x) / float(self.stride_shape[1]))
            self.output_tensor = np.zeros((b, self.num_kernels, int(out_height), int(out_width)))

            if (pad_height % 2 == 0):
                pad_top = pad_height / 2
                pad_bottom = pad_height - pad_top
            else:
                pad_top = np.floor(pad_height / 2)
                pad_bottom = pad_top + 1
            if (pad_width % 2 == 0):
                pad_left = pad_width / 2
                pad_right = pad_width - pad_left
            else:
                pad_left = np.floor(pad_width / 2)
                pad_right = pad_left + 1
            padded_input = np.zeros((b, self.c, y + pad_height, x + pad_width))

            for batch in range(b):
                for channel in range(self.c):
                    padded_input[batch, channel] = np.pad(input_tensor[batch, channel],
                                                          [(int(pad_top), int(pad_bottom)),
                                                           (int(pad_left), int(pad_right))], mode='constant')

            self.padded_input = padded_input

            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    for h in range(int(out_height)):
                        h_start = h * self.stride_shape[0]
                        h_end = h_start + self.m
                        for w in range(int(out_width)):
                            w_start = w * self.stride_shape[1]
                            w_end = w_start + self.n

                            subset = padded_input[batch, :, h_start:h_end, w_start:w_end]
                            self.output_tensor[batch, out_channel, h, w] = np.sum(
                                subset * self.weights[out_channel, :, :, :] + self.bias[out_channel])



        else:
            pad_height = np.ceil((self.stride_shape[0] * y - self.stride_shape[0] + self.m - y))
            out_height = np.ceil(float(y) / float(self.stride_shape[0]))
            self.output_tensor = np.zeros((b, self.num_kernels, int(out_height)))
            if (pad_height % 2 == 0):
                pad_top = pad_height / 2
                pad_bottom = pad_top
            else:
                pad_top = np.floor(pad_height / 2)
                pad_bottom = pad_top + 1
            padded_input = np.zeros((b, self.c, int(y + pad_height)))
            for batch in range(b):
                for channel in range(self.c):
                    padded_input[batch, channel] = np.pad(input_tensor[batch, channel],
                                                          [(int(pad_top), int(pad_bottom))], mode='constant')

            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    for h in range(int(out_height)):
                        subset = padded_input[batch, :, h * self.stride_shape[0]:h * self.stride_shape[0] + self.m]
                        self.output_tensor[batch, out_channel, h] = np.sum(
                            subset * self.weights[out_channel, :, :] + self.bias[out_channel])

        return self.output_tensor

    def backward(self, error_tensor):
        self.prev_error = np.zeros(self.input_tensor.shape)

        num_of_batch = self.input_tensor.shape[0]
        num_of_channels = error_tensor.shape[1]
        height = error_tensor.shape[2]

        if (self.img_2d == True):
            error_tensor_strided = error_tensor[:, :, :: self.stride_shape[0], :: self.stride_shape[1]]
            width = error_tensor.shape[3]
            self.gradient_w = np.zeros(num_of_batch, num_of_channels, height, width)

            for batch in range(num_of_batch):
                temp_image = self.input_tensor[batch]
                for channel in range(num_of_channels):
                    padding_size, pad_top, pad_bottom, pad_left, pad_right = self.determine_padding_size()

                    padded_input = np.zeros(self.input_cha_num * int(self.input_tensor[2] + padding_size) * int(self.input_tensor[3] + padding_size))
                    padded_input = np.reshape(padded_input, (self.input_cha_num * int(self.input_tensor[2] + padding_size) * int(self.input_tensor[3] + padding_size)))

                    for temp_channel in range(self.input_cha_num):
                        padded_input = np.pad(self.input_tensor[batch, temp_channel], [(int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))],
                                                mode='constant')

                    self.gradient_w[batch, channel] = signal.correlate(padded_input, error_tensor[channel], mode="valid")[0]
        else:
            error_tensor_strided = error_tensor[:, :, :: self.stride_shape[0]]
            self.gradient_w = np.zeros(num_of_batch * num_of_channels * height)
            self.gradient_w = np.reshape(self.gradient_w, (num_of_batch, num_of_channels, height))

            for batch in range(num_of_batch):
                for channel in range(num_of_channels):
                    padding_size, pad_top, pad_bottom = self.determine_padding_size()

                    padded_input = np.zeros(self.input_cha_num * int(self.input_tensor.shape[2] + padding_size))
                    padded_input = np.reshape(padded_input, (self.input_cha_num, self.input_tensor.shape[2] + padding_size))

                    for temp_channel in range(self.input_cha_num):
                        padded_input[temp_channel] = np.pad(self.input_tensor[batch, temp_channel],
                                              [(int(pad_top), int(pad_bottom))],
                                              mode='constant')

                    self.gradient_w[batch, channel] = \
                    signal.correlate(padded_input, error_tensor_strided[channel], mode="valid")[0]
        return self.prev_error

    def determine_padding_size(self):
        if (self.img_2d == True):
            if ((self.n % 2) == 0):
                padding_size = self.n / 2
            else:
                padding_size = self.n // 2

            if (padding_size % 2 == 0):
                pad_top = padding_size / 2
                pad_bottom = padding_size / 2
                pad_left = padding_size / 2
                pad_right = padding_size / 2
            else:
                pad_top = padding_size // 2
                pad_left = padding_size // 2

                pad_right = padding_size - pad_top
                pad_bottom = padding_size - pad_left

            return (padding_size, pad_top, pad_bottom, pad_left, pad_right)

        else:
            if ((self.m % 2) == 0):
                padding_size = self.m / 2
            else:
                padding_size = self.m // 2

            if (padding_size % 2 == 0):
                pad_top = padding_size / 2
                pad_bottom = padding_size / 2
            else:
                pad_top = padding_size // 2
                pad_bottom = padding_size - pad_top

            return (padding_size, pad_top, pad_bottom)
