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
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.opt_w = None
        self.opt_b = None

        self.img_2d = False

        self._optimizer_b = None
        self._optimizer_w=None

        self.input_cha_num = self.convolution_shape[0]

        self.m = self.convolution_shape[1]

        # in case of 2D input
        if (len(convolution_shape) == 3):
            self.img_2d = True
            self.n = self.convolution_shape[2]
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_cha_num, self.m, self.n)))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
        else:
            self.weights = np.random.uniform(0, 1, ((self.num_kernels, self.input_cha_num, self.m)))
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
        if (self.img_2d == True):
            x = input_tensor.shape[3]
            self.output_tensor = np.zeros((b, self.num_kernels, y, x))
        else:
            self.output_tensor = np.zeros((b, self.num_kernels, y))

        if (self.img_2d == True):
            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    self.output_tensor[batch, out_channel] = signal.correlate(self.find_padded_input(self.input_tensor)[batch],
                                                                              self.weights[out_channel], mode='valid')[0] + self.bias[out_channel]
            self.output_tensor = self.output_tensor[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]

        else:
            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    self.output_tensor[batch, out_channel] = signal.correlate(self.find_padded_input(self.input_tensor)[batch],
                                                                              self.weights[out_channel], mode='valid')[0] + self.bias[out_channel]
            self.output_tensor = self.output_tensor[:, :, ::self.stride_shape[0]]

        return self.output_tensor

    def backward(self, error_tensor):
        self.prev_error = np.zeros(self.input_tensor.shape)
        self.gradient_w = np.zeros(self.weights.shape)

        if (self.img_2d == True):
            """
            1. gradient w.r.t. bias
            """
            self.gradient_b = np.zeros(self.bias.shape)
            for ker in range(self.num_kernels):
                self.gradient_b[ker] = np.sum(error_tensor[:, ker, :, :])
            """
            2. gradient w.r.t. previous layer
            """
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
            # new_weights[:,:] = np.flip(new_weights[:,:], axis= 0)
            for in_ch in range(self.input_cha_num):
                for ker in range(self.num_kernels):
                    new_weights[in_ch, ker] = np.flip(new_weights[in_ch, ker], axis=0)

            # do padding+convolution
            for batch in range(error_tensor.shape[0]):
                for ker in range(self.input_cha_num):
                    self.prev_error[batch, ker] = \
                    signal.convolve(self.find_padded_input(upsampled_error)[batch], np.rot90(np.rot90(new_weights[ker])), mode='valid')[0]
            """
            3. gradient w.r.t weights
            """
            self.gradient_w = self.find_gradient_weights(upsampled_error)
            """
            4. update weight & bias optimizer
            """
            if (self._optimizer_b != None):
                self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_b)
            if (self._optimizer_w != None):
                self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_w)

        else:
            """1. gradient w.r.t. bias """
            self.gradient_b = np.zeros(self.bias.shape)
            for ker in range(self.num_kernels):
                self.gradient_b[ker] = np.sum(error_tensor[:, ker, :])
            """
            2. gradient w.r.t previous layer
            """
            self.prev_error = np.zeros(
                (error_tensor.shape[0], self.input_cha_num, self.input_tensor.shape[2]))

            upsampled_error = self.find_upsampled_error(error_tensor)

            # rearrage kernel
            new_weights = np.zeros((self.input_cha_num, self.num_kernels, self.m))
            for in_ch in range(self.input_cha_num):
                temp = np.zeros((self.num_kernels, self.m))
                for ker in range(self.num_kernels):
                    temp[ker] = self.weights[ker, in_ch]
                new_weights[in_ch] = temp
            # flip spatial space
            # new_weights[:,:] = np.flip(new_weights[:,:], axis= 0)
            for in_ch in range(self.input_cha_num):
                for ker in range(self.num_kernels):
                    new_weights[in_ch, ker] = np.flip(new_weights[in_ch, ker], axis=0)

            # do padding+convolution
            for batch in range(error_tensor.shape[0]):
                for ker in range(self.input_cha_num):
                    self.prev_error[batch, ker] = \
                        signal.convolve(self.find_padded_input(upsampled_error)[batch],
                                        np.rot90(np.rot90(new_weights[ker])), mode='valid')[0]
            """
            3.gradient w.r.t. weights
            """
            self.gradient_w = self.find_gradient_weights(upsampled_error)
            """
            4. update bias & weight optimizer
            """
            if (self._optimizer_b != None):
                self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_b)
            if (self._optimizer_w != None):
                self.weights = self._optimizer_w.calculate_update(self.weights, self.gradient_w)


        return self.prev_error

    def find_padded_input(self, input_tensor):
        if (self.img_2d == True):
            pad_height, pad_width, pad_top, pad_bottom, pad_left, pad_right = self.determine_padding_size()
            array_temp = np.zeros((input_tensor.shape[0], input_tensor.shape[1],
                                   input_tensor.shape[2] + pad_height, input_tensor.shape[3] + pad_width))
            array_temp = np.pad(input_tensor,
                                [(0, 0), (0, 0), (int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))],
                                mode='constant')
            return array_temp
        else:
            pad_height, pad_top, pad_bottom = self.determine_padding_size()
            array_temp = np.zeros(
                (input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2] + pad_height))
            array_temp = np.pad(input_tensor, [(0, 0), (0, 0), (int(pad_top), int(pad_bottom))], mode='constant')
            return array_temp

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
            padded_input = np.pad(self.input_tensor, [(0,0), (0,0), (int(pad_top), int(pad_bottom)), (int(pad_first), int(pad_second))],
                                  mode='constant')

            temp_gradient_w = np.zeros((self.num_kernels, self.input_cha_num,
                                         np.abs(len(padded_input[0, 0]) - len(upsampled_error[0, 0])) + 1,
                                         np.abs(len(padded_input[0, 0, 0]) - len(upsampled_error[0, 0, 0])) + 1))
            for ker in range(self.num_kernels):
                for in_ch in range(self.input_cha_num):
                    temp = signal.correlate(padded_input[:, in_ch], upsampled_error[:, ker], mode='valid')[0]
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
            padded_input = np.pad(self.input_tensor,[(0,0), (0,0), (int(pad_top), int(pad_bottom))], mode='constant')

            temp_gradient_w = np.zeros((self.num_kernels, self.input_cha_num,
                                        np.abs(len(padded_input[0, 0]) - len(upsampled_error[0, 0])) + 1))
            for ker in range(self.num_kernels):
                for in_ch in range(self.input_cha_num):
                    temp = signal.correlate(padded_input[:, in_ch], upsampled_error[:, ker], mode='valid')[0]
                    temp_gradient_w[ker, in_ch] += temp

        return temp_gradient_w

    def find_upsampled_error(self, error_tensor):
        if (self.img_2d):
            upsampled_error = np.zeros(
                (error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2], self.input_tensor.shape[3]))

            for r in range(error_tensor.shape[2]):
                if (r * self.stride_shape[0] < upsampled_error.shape[2]):
                    for c in range(error_tensor.shape[3]):
                        if (c * self.stride_shape[1] < upsampled_error.shape[3]):
                            upsampled_error[:, :, r * self.stride_shape[0], c * self.stride_shape[1]] = error_tensor[:, :, r, c]
                        else:
                            print("not found")
        else:
            upsampled_error = np.zeros(
                (error_tensor.shape[0], self.num_kernels, self.input_tensor.shape[2]))

            for r in range(error_tensor.shape[2]):
                if (r * self.stride_shape[0] < upsampled_error.shape[2]):
                    upsampled_error[:, :, r * self.stride_shape[0]] = error_tensor[:, :, r]
                else:
                    print("not found")

        return upsampled_error
