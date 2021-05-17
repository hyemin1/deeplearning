from Layers import Base
import numpy as np
import math
from scipy import ndimage
from scipy import signal
from Layers import Initializers
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
            self.bias = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m * self.n))
            self.bias = np.reshape(self.bias, (self.num_kernels, self.input_cha_num, self.m, self.n))
        #in case of 1D input
        else:
            # initialize weights using given kernel shape
            self.weights=np.random.uniform(0,1,(self.num_kernels*self.input_cha_num*self.m))
            self.weights=np.reshape(self.weights,(self.num_kernels,self.input_cha_num,self.m))
            #initialize bias
            self.bias = np.random.uniform(0, 1, (self.num_kernels * self.input_cha_num * self.m))

    @property
    def gradient_weights(self):
        return self.gradient_w
    @property
    def gradient_bias(self):
        return self.gradient_b
    @property
    def optimizer(self,opt):
        self.opt_w=np.copy.deepcopy(opt)
        self.opt_b=np.copy.deepcopy(opt)

    #should be corrected
    def initialize(self,weights_initializer,bias_initializer):

        if(self.img_2d==True):
            self.fan_in = self.input_cha_num*self.m*self.n
            self.fan_out = self.num_kernels*self.m*self.n

            weights_initializer.initialize((self.m,self.n),self.fan_in,self.fan_out)
            bias_initializer.initialize((self.m,self.n),self.fan_in,self.fan_out)
        else:
            self.fan_in = self.input_cha_num * self.m
            self.fan_out = self.num_kernels * self.m

            weights_initializer.initialize((self.m), self.fan_in, self.fan_out)
            bias_initializer.initialize((self.m), self.fan_in, self.fan_out)

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
        if(self.img_2d==True):
            pad_height= np.ceil((self.stride_shape[0]*y-self.stride_shape[0]+self.m-y)/2)
            pad_height = int(pad_height)
            pad_width = np.ceil((self.stride_shape[1]*x-self.stride_shape[1]+self.n-x)/2)
            pad_width = int(pad_width)

            #padded_input = np.pad(input_tensor,[(0,0),(pad_height,pad_height),(pad_width,pad_width),(0,0)],mode='constant',constant_values=0)

            out_height = np.ceil(float(y) / float(self.stride_shape[0]))
            out_width = np.ceil(float(x) / float(self.stride_shape[1]))
            self.output_tensor = np.zeros((b,self.num_kernels,int(out_height),int(out_width)))
            if (pad_height % 2 == 0):
                pad_top = pad_height / 2
                pad_bottom = pad_top
            else:
                pad_top = np.floor(pad_height / 2)
                pad_bottom = pad_top + 1
            if (pad_width % 2 == 0):
                pad_left = pad_width / 2
                pad_right = pad_left
            else:
                pad_left = np.floor(pad_width / 2)
                pad_right = pad_left + 1

            #pad_in = np.pad(input_tensor,[(0,0),(0,0),(int(pad_top),int(pad_bottom)),(int(pad_left),int(pad_right))],mode='constant',constant_values=0)
            #for batch in range(b):
            #    for out_channel in range(self.num_kernels):
            #        sum=0
            #        for in_channel in range(self.input_cha_num):
            #            #sum=sum+ np.sum(signal.convolve2d(input_tensor[batch][in_channel],self.weights[batch][out_channel],mode='same') )
            #            self.output_tensor[batch][out_channel]=signal.convolve2d(input_tensor[batch][in_channel],self.weights[batch][out_channel],mode='same')
            #for batch in range(b):
            #    for out_ch in range(self.num_kernels):
            #        for in_ch in range(self.input_cha_num):

            #for n in range(b):
            #    for i in range(int(out_height)):
            #        for j in range(int(out_width)):
            #            for ou_ch in range(self.num_kernels):
            #                self.output_tensor[n,ou_ch, i, j] = np.multiply(
            #                    input_tensor[n, :,i:i + self.m, j:j + self.n], self.weights[:, ou_ch,:, :]).sum()

           # for x in range(out_height):
            #    for y in range(out_width):
            #        for i in range(self.m):
            #            for j in range(self.n):
            #                self.output_tensor[x][y] += pad_in[x + i][y + j] * self.weights[i][j]

            # for batch in range(b):
            #     for out_ch in range(self.num_kernels):
            #         for in_ch in range(self.input_cha_num):
            #             self.output_tensor[batch][out_ch]+=signal.convolve2d(pad_in[batch][in_ch],self.weights[out_ch][in_ch],mode='valid')



        else:
            pad_height = np.ceil((self.stride_shape[0] * y - self.stride_shape[0] + self.m - y) / 2)
            out_height = np.ceil(float(y) / float(self.stride_shape[0]))
            self.output_tensor = np.zeros((b, self.num_kernels, int(out_height)))
            if (pad_height % 2 == 0):
                pad_top = pad_height / 2
                pad_bottom = pad_top
            else:
                pad_top = np.floor(pad_height / 2)
                pad_bottom = pad_top + 1
        #set padding size



        return self.output_tensor


    def backward(self,error_tensor):
        if(self.img_2d==True):
            print()
            #self.gradient_b=np.sum(error_tensor,axis=(0,1),keepdims=True)
        else:
            print()
            #self.gradient_b = np.sum(error_tensor, axis=(1, 2), keepdims=True)
        return

