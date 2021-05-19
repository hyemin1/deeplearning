from Layers import Base
import numpy as np
import math
from scipy import ndimage
from scipy import signal
from Layers import Initializers
from PIL import Image
import copy
class Conv(Base.BaseLayer):
    def __init__(self,stride_shape,convolution_shape,num_kernels):
        super().__init__()
        self.trainable=True
        self.weights=None
        self.bias=None
        self.stride_shape=stride_shape
        self.convolution_shape = convolution_shape#shape for convolution
        self.num_kernels = num_kernels #total number of kernels
        #optimizer to update wdight
        self.opt_w=None
        #optimizer to update bias
        self.opt_b=None
        self.img_2d=False

        #number of input channels
        self.input_cha_num=self.convolution_shape[0]

        #in 1D: length of array, in 2D: height of matrix
        self.m=self.convolution_shape[1]

        #in case of 2D input
        if (len(convolution_shape)==3):
            self.img_2d=True
            #width of matrix
            self.n=self.convolution_shape[2]
            #initialize weights using given kernel shape
            self.weights=np.random.uniform(0,1,(self.num_kernels*self.input_cha_num*self.m*self.n))
            self.weights=np.reshape(self.weights,(self.num_kernels,self.input_cha_num,self.m,self.n))
            #initialize bias
            #self.bias = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m * self.n))
            #self.bias = np.reshape(self.bias, (self.num_kernels, self.input_cha_num, self.m, self.n))
            self.bias= np.random.uniform(0,1,(self.num_kernels))
            #self.bias = np.reshape(self.bias,(self.num_kernels,self.input_cha_num))
        #in case of 1D input
        else:
            # initialize weights using given kernel shape
            self.weights=np.random.uniform(0,1,(self.num_kernels*self.input_cha_num*self.m))
            self.weights=np.reshape(self.weights,(self.num_kernels,self.input_cha_num,self.m))
            #initialize bias
            #self.bias = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m))
            self.bias = np.random.uniform(0, 1, (self.num_kernels))
            #self.bias = np.reshape(self.bias, (self.num_kernels, self.input_cha_num))

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
    def optimizer(self,opt):
        self._optimizer=copy.deepcopy(opt)

    #should be corrected
    def initialize(self,weights_initializer,bias_initializer):

        if(self.img_2d==True):
            self.fan_in = self.input_cha_num*self.m*self.n
            self.fan_out = self.num_kernels*self.m*self.n

            self.weights=weights_initializer.initialize(self.weights.shape,self.fan_in,self.fan_out)
            self.bias=bias_initializer.initialize(self.bias.shape,self.fan_in,self.fan_out)
        else:
            self.fan_in = self.input_cha_num * self.m
            self.fan_out = self.num_kernels * self.m

            self.weights=weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
            self.bias=bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

    def forward(self,input_tensor):
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
        self.input_tensor=input_tensor
        #number of batches
        b=input_tensor.shape[0]
        #set dimension(number of input channels per batch)
        self.c= input_tensor.shape[1]
        #spatial dimension
        #in 1D: length of array, in 2D: height of matrix
        y = input_tensor.shape[2]
        #in 2D
        if(self.img_2d==True):
            #width of matrix
            x = input_tensor.shape[3]

        """
        1. calculate the output tensor shape
        2. do zero padding
        """
        if (self.img_2d == True):
            # width of matrix
            x = input_tensor.shape[3]

        """
        1. calculate the output tensor shape
        2. do zero padding
        """
        if (self.img_2d == True):
            self.output_tensor=np.zeros((b,self.num_kernels,y,x))
            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    temp=self.input_tensor[batch]
                    pad_height=self.m-1
                    pad_width=self.n-1

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

                    temp2 = np.zeros(
                        (self.input_cha_num, int(temp.shape[1] + pad_height), int(temp.shape[2] + pad_width)))
                    for channel in range(temp.shape[0]):
                        temp2[channel] = np.pad(temp[channel],
                                                [(int(pad_top), int(pad_bottom)), (int(pad_left), int(pad_right))],
                                                mode='constant')
                    self.output_tensor[batch, out_channel] = signal.correlate(temp2, self.weights[out_channel], mode='valid')[0] + self.bias[out_channel]

            self.output_tensor=self.output_tensor[:,:,::self.stride_shape[0],::self.stride_shape[1]]

        else:
            self.output_tensor = np.zeros((b, self.num_kernels, y))
            for batch in range(b):
                for out_channel in range(self.num_kernels):
                    temp=self.input_tensor[batch]
                    pad_height = self.m - 1
                    if (pad_height % 2 == 0):
                        pad_top = pad_height / 2
                        pad_bottom = pad_height - pad_top
                    else:
                        pad_top = np.floor(pad_height / 2)
                        pad_bottom = pad_top + 1
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


    def backward(self,error_tensor):
        self.prev_error = np.zeros(self.input_tensor.shape)
        self.gradient_w=np.zeros(self.weights.shape)
        
        if(self.img_2d==True):
            self.gradient_b=np.zeros(self.bias.shape)
            new_weights = np.zeros(self.weights.shape)
            #self.gradient_b=np.sum(error_tensor, axis=(1, 2, 3), keepdims=True)
            # for batch in range(error_tensor.shape[0]):
            #     for ker in range(error_tensor.shape[1]):
            #         for h in range(error_tensor.shape[2]):
            #             for w in range(error_tensor.shape[3]):
            #                 self.gradient_b[ker]+=error_tensor[batch,ker,h,w]
            #print(error_tensor.shape)
            #print(self.weights.shape)
            #new_weights[batch,ker]=np.rot90(np.rot90(self.weights[]))
            #new_weights = np.rot90(np.rot90(self.weights))
            new_weights = np.reshape(new_weights,(self.weights.shape[1],self.weights.shape[0],self.weights.shape[2],self.weights.shape[3]))
            # for ker in range(self.weights.shape[0]):
            #     for in_cha in range(self.weights.shape[1]):
            #         new_weights[ker,in_cha]=np.fliplr(self.weights[ker,in_cha])
            # new_weights=new_weights.flatten()

            #reshaping needed
            #new_weights.reshape(new_weights,int(self.weights.shape[1]),int(self.weights.shape[0]),int(self.weights.shape[2]),int(self.weights.shape[3]))
            #print(new_weights.shape)
            #new_weights=np.reshape(new_weights,(error_tensor.shape[0],self.input_cha_num,error_tensor.shape[2],error_tensor.shape[3]))

            #print(new_weights.shape)

        else:
            print()
            #self.gradient_b = np.sum(error_tensor, axis=(1, 2), keepdims=True)
            #self.gradient_b = np.sum(error_tensor, axis=0)
            self.gradient_b=np.zeros(self.bias.shape)
            for batch in range(error_tensor.shape[0]):
                for ker in range(error_tensor.shape[1]):
                    for h in range(error_tensor.shape[2]):
                            self.gradient_b[ker]+=error_tensor[batch,ker,h]
        return self.prev_error



