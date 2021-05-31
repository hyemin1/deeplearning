from Layers import Base
import numpy as np
from scipy import signal
import copy

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

        self._optimizer_b = None
        self._optimizer_w=None

        # number of input channels
        self.input_cha_num = self.convolution_shape[0]

        # in 1D: length of array, in 2D: height of matrix
        self.m = self.convolution_shape[1]

        # in case of 2D input
        if (len(convolution_shape) == 3):
            self.img_2d = True
            self.n = self.convolution_shape[2]
            self.weights = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m * self.n))
            self.weights = np.reshape(self.weights, (self.num_kernels, self.input_cha_num, self.m, self.n))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
        else:
            self.weights = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m))
            self.weights = np.reshape(self.weights, (self.num_kernels, self.input_cha_num, self.m))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))

    @property
    def gradient_weights(self):
        return self.gradient_w

    @property
    def gradient_bias(self):
        return self.gradient_b

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer_b = copy.deepcopy(opt)
        self._optimizer_w=copy.deepcopy(opt)

    # should be corrected
    def initialize(self, weights_initializer, bias_initializer):

        if (self.img_2d == True):
            self.fan_in = self.input_cha_num * self.m * self.n
            self.fan_out = self.num_kernels * self.m * self.n

            self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
            self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)
        else:
            self.fan_in = self.input_cha_num * self.m
            self.fan_out = self.num_kernels * self.m

            self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
            self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        b = input_tensor.shape[0]
        self.c = input_tensor.shape[1]
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
            self.output_tensor = np.zeros((b, self.num_kernels, y, x))
            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    temp = self.input_tensor[batch]
                    pad_height, pad_width, pad_top, pad_bottom, pad_left, pad_right = self.determine_padding_size()
                    temp2 = np.zeros(
                        (self.input_cha_num, int(temp.shape[1] + pad_height), int(temp.shape[2] + pad_width)))
                    for channel in range(temp.shape[0]):
                        temp2[channel] = np.pad(temp[channel],
                                                [(int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))],
                                                mode='constant')
                    self.output_tensor[batch, out_channel] = signal.correlate(temp2, self.weights[out_channel], mode='valid')[0] + self.bias[out_channel]

            self.output_tensor = self.output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        else:
            self.output_tensor = np.zeros((b, self.num_kernels, y))
            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    temp = self.input_tensor[batch]
                    pad_height, pad_top, pad_bottom = self.determine_padding_size()
                    temp2 = np.zeros(
                        (self.input_cha_num, int(temp.shape[1] + pad_height)))
                    for channel in range(temp.shape[0]):
                        temp2[channel] = np.pad(temp[channel],
                                                [(int(pad_top), int(pad_bottom))],
                                                mode='constant')
                    self.output_tensor[batch, out_channel] = signal.correlate(temp2, self.weights[out_channel], mode='valid')[
                                                         0] + self.bias[out_channel]

            self.output_tensor = self.output_tensor[:, :, ::self.stride_shape[0]]
        return self.output_tensor

    def backward(self, error_tensor):
        self.prev_error = np.zeros(self.input_tensor.shape)
        self.gradient_w = np.zeros(self.weights.shape)

        if (self.img_2d == True):

            self.gradient_b = np.zeros(self.bias.shape)
            for ker in range(self.num_kernels):
                self.gradient_b[ker] = np.sum(error_tensor[:, ker, :, :])

            self.prev_error = np.zeros(
                (error_tensor.shape[0], self.input_cha_num, self.input_tensor.shape[2], self.input_tensor.shape[3]))

            upsampled_error = self.find_upsampled_error(error_tensor)

            # rearrage kernel
            new_weights = np.zeros((self.input_cha_num, self.num_kernels, self.m, self.n))
            for in_ch in range(self.input_cha_num):
                temp = np.zeros((self.num_kernels, self.m, self.n))
                for ker in range(self.num_kernels):
                    temp[ker] = self.weights[ker, in_ch]
                new_weights[in_ch] = temp
            # flip spatial space
            for in_ch in range(self.input_cha_num):
                for ker in range(self.num_kernels):
                    new_weights[in_ch, ker] = np.flip(new_weights[in_ch, ker], axis=0)

            # do padding+convolution
            for batch in range(error_tensor.shape[0]):
                for ker in range(self.input_cha_num):
                    temp = upsampled_error[batch]
                    pad_height, pad_width, pad_top, pad_bottom, pad_left, pad_right = self.determine_padding_size()
                    temp2 = np.zeros(
                        (self.num_kernels, int(temp.shape[1] + pad_height), int(temp.shape[2] + pad_width)))
                    for channel in range(temp.shape[0]):
                        temp2[channel] = np.pad(temp[channel],
                                                [(int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))],
                                                mode='constant')
                    self.prev_error[batch, ker] = \
                    signal.convolve(temp2, np.rot90(np.rot90(new_weights[ker])), mode='valid')[0]

            self.gradient_w = self.find_gradient_weights(upsampled_error)
            if (self._optimizer_b != None):
                self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_b)
            if (self._optimizer_w != None):
                self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_w)

        else:
            upsampled_error = self.find_upsampled_error(error_tensor)
            self.gradient_w = self.find_gradient_weights(upsampled_error)

        return self.prev_error

    def determine_padding_size(self):
        if (self.img_2d):
            pad_height = self.m - 1
            pad_width = self.n - 1

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
            return (pad_height, pad_width, pad_top, pad_bottom, pad_left, pad_right)
        else:
            pad_height = self.m - 1
            if (pad_height % 2 == 0):
                pad_top = pad_height / 2
                pad_bottom = pad_height - pad_top
            else:
                pad_top = np.floor(pad_height / 2)
                pad_bottom = pad_top + 1
            return (pad_height, pad_top, pad_bottom)


    def find_gradient_weights(self, upsampled_error):
        if (self.img_2d):
            pad_size = self.weights.shape[3] - 1
            total = self.input_tensor.shape[3] + pad_size

            pad_height = self.weights.shape[3] - 1
            if (self.stride_shape[0] > self.stride_shape[1]):
                pad_height = pad_size - self.stride_shape[0]
            total_2 = self.input_tensor.shape[2] + pad_height
            if (pad_size % 2 == 0):
                pad_first = pad_size / 2
                pad_second = pad_first
            else:
                pad_first = np.floor(pad_size / 2)
                pad_second = pad_first + 1

            if (pad_height % 2 == 0):
                pad_top = pad_height / 2
                pad_bottom = pad_top
            else:
                pad_top = np.floor(pad_height / 2)
                pad_bottom = pad_top + 1
            padded_input = np.zeros((self.input_tensor.shape[0], self.input_cha_num, total_2, total))

            for batch in range(self.input_tensor.shape[0]):
                for in_ch in range(self.input_cha_num):
                    padded_input[batch, in_ch] = np.pad(self.input_tensor[batch, in_ch],
                                                        [(int(pad_top), int(pad_bottom)),
                                                         (int(pad_first), int(pad_second))], mode='constant')

            temp_gradient_w = np.zeros((self.num_kernels, self.input_cha_num,
                                         np.abs(len(padded_input[0, 0]) - len(upsampled_error[0, 0])) + 1,
                                         np.abs(len(padded_input[0, 0, 0]) - len(upsampled_error[0, 0, 0])) + 1))

            for batch in range(self.input_tensor.shape[0]):
                for ker in range(self.num_kernels):
                    for in_ch in range(self.input_cha_num):
                        temp = signal.correlate2d(padded_input[batch, in_ch], upsampled_error[batch, ker], mode='valid')
                        temp_gradient_w[ker, in_ch] += temp

        else:
            pad_size = self.weights.shape[2] - 1
            total = self.input_tensor.shape[2] + pad_size

            if (pad_size % 2 == 0):
                pad_top = pad_size / 2
                pad_bottom = pad_top
            else:
                pad_top = np.floor(pad_size / 2)
                pad_bottom = pad_top + 1

            padded_input = np.zeros((self.input_tensor.shape[0], self.input_cha_num, total))

            for batch in range(self.input_tensor.shape[0]):
                for in_ch in range(self.input_cha_num):
                    padded_input[batch, in_ch] = np.pad(self.input_tensor[batch, in_ch],
                                                        [(int(pad_top), int(pad_bottom))], mode='constant')

            temp_gradient_w = np.zeros((self.num_kernels, self.input_cha_num,
                                        np.abs(len(padded_input[0, 0]) - len(upsampled_error[0, 0])) + 1))

            for batch in range(self.input_tensor.shape[0]):
                for ker in range(self.num_kernels):
                    for in_ch in range(self.input_cha_num):
                        temp = signal.correlate(padded_input[batch, in_ch], upsampled_error[batch, ker], mode='valid')
                        temp_gradient_w[ker, in_ch] += temp


        return temp_gradient_w


    def find_upsampled_error(self, error_tensor):
        if (self.img_2d):
            upsampled_error = np.zeros(
                (error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))

            for batch in range(self.input_tensor.shape[0]):
                for ker in range(self.num_kernels):
                    for r in range(error_tensor.shape[2]):
                        if (r * self.stride_shape[0] < upsampled_error.shape[2]):
                            for c in range(error_tensor.shape[3]):
                                if (c * self.stride_shape[1] < upsampled_error.shape[3]):
                                    upsampled_error[batch, ker, r * self.stride_shape[0], c * self.stride_shape[1]] = error_tensor[batch, ker, r, c]
                                else:
                                    print("not found")
        else:
            upsampled_error = np.zeros(
                (error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2]))

            for batch in range(self.input_tensor.shape[0]):
                for ker in range(self.num_kernels):
                    for r in range(error_tensor.shape[2]):
                        if (r * self.stride_shape[0] < upsampled_error.shape[2]):
                            upsampled_error[batch, ker, r * self.stride_shape[0]] = \
                                error_tensor[batch, ker, r]
                        else:
                            print("not found")

        return upsampled_error
