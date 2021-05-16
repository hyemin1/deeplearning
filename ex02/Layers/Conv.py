from Layers import Base
import numpy as np
import math
from scipy import ndimage
class Conv(Base.BaseLayer):
    def __init__(self,stride_shape,convolution_shape,num_kernels):
        super().__init__()
        self.trainable=True
        self.weights=None
        self.bias=None
        self.stride_shape=stride_shape
        self.convolution_shape = convolution_shape#shape for convolution
        self.num_kernels = num_kernels #total number of kernels
        self.opt_1=None
        self.opt_2=None
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
        #in case of 1D input
        else:
            # initialize weights using given kernel shape
            self.weights=np.random.uniform(0,1,(self.num_kernels*self.input_cha_num*self.m))
            self.weights=np.reshape(self.weights,(self.num_kernels,self.input_cha_num,self.m))
        self.bias=0
    @property
    def gradient_weights(self):
        return self.gradient_w
    @property
    def gradient_bias(self):
        return self.gradient_b
    @property
    def optimizer(self,opt):
        self.opt_1=opt
        self.opt_2=opt

    #should be corrected
    def initialize(self,weights_initializer,bias_initializer):
        fan_in = len(self.input_tensor.shape[1])*self.convolution_shape.shape[0]*self.convolution_shape.shape[1]
        #fan_out =

        self.weights = weights_initializer.initialize(self.weights.shape,fan_in,fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape,fan_in,fan_out)

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
            pad_height= (self.m-1)/2
            pad_width = (self.n-1)/2

            output_height = math.floor(((y-self.m+2*pad_height)/self.stride_shape[0])+1)
            output_width =math.floor(((x-self.n+2*pad_width)/self.stride_shape[1])+1)
            self.output_tensor = np.zeros((b,self.num_kernels,output_height,output_width))

            #self.padded_input = np.pad(input_tensor, ((pad_height, pad_height), (pad_width, pad_width)), 'same',constant_values=0)
        else:
            pad=(self.m-1)/2
            output_length = (y-self.m+2*pad)/self.stride_shape[0]+1
            self.output_tensor=np.zeros((self.num_kernels,self.c,output_length))
            #self.padded_input=np.pad(input_tensor,pad,'same',constant_values=0)
        return self.output_tensor
    """
    Adding zero padding
    """
        padding_size = (0,0)
        if self.padding == "same":
            height, width = input_tensor.shape[2], input_tensor.shape[3]
            padding_size = (height - 1) // 2, (width - 1) // 2
        elif self.padding == "valid":
            padding_size = (0,0)
        else:
            print("Invalid padding!")

        padding_added_tensor = np.pad(array= input_tensor,
                                      pad_width= ((0, 0), (padding_size[0], padding_size[0]), (padding_size[1], padding_size[1]), (0, 0)),
                                      mode = 'constant')

        """
        3. do convolution using scipy library,padded input,kernel,bias
        """
"""
        for batch in range(b):
            for output_channel in range(self.num_kernels):
                per_channel_temp=0
                for input_channel in range(self.input_cha_num):
                    temp=ndimage.convolve(input_tensor[batch][input_channel],self.weights[output_channel][input_channel],mode='constant',cval=1)
                    temp=temp+self.bias
                    per_channel_temp+=temp
                    #np.append(per_channel_temp,temp)
                self.output_tensor[batch][output_channel]=per_channel_temp
"""



