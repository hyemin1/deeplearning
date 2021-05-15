from Layers import Base
import numpy as np
class Conv(Base.BaseLayer):
    def __init__(self,stride_shape,convolution_shape,num_kernels):
        super().__init__()
        self.trainable=True
        self.weights=None
        self.bias=None
        self.stride_shape=stride_shape
        self.convolution_shape = convolution_shape#kernel shape
        self.num_kernels = num_kernels
        self.opt_1=None
        self.opt_2=None

        #number of channels per batch
        self.channel_num=self.convolution_shape[0]

        self.m=self.convolution_shape[1]
        #in case of 2D input
        if (len(convolution_shape)==3):
            self.n=self.convolution_shape[2]
            #initialize weights using given kernel shape
            self.weights=np.randpm.uniform(0,1,(self.channel_num*self.m*self.n))
            self.weights=np.reshape(self.weights,(self.channel_num,self.m,self.n))
        #in case of 1D input
        else:
            # initialize weights using given kernel shape
            self.weights=np.random.uniform(0,1,(self.channel_num*self.m))
            self.weights=np.reshape(self.weights,(self.channel_num,self.m))
        self.bias=None
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
        1. do zero padding
        2. calculate the output tensor shape
        3. do convolution using scipy library,weight matrix,bias
        """

        """
        0. set input layout
        """
        #set dimension
        self.dimen=len(input_tensor.shape)-2
        #number of batches
        b=input_tensor.shape[0]
        #number of channels
        c= input_tensor.shape[1]
        #spatial dimension
        y = input_tensor.shape[2]
        if(self.dimen==2):
            x = input_tensor.shape[3]

        """
        1. do zero padding 
        2. calculate the output tensor shape
        """
        #maybe wrong
        #case of 1*1 convolutions should be added
        if (self.dimen==2):
            pad_y = y - self.m
            pad_x = x - self.n
            if (pad_x != 0 or pad_y != 0):
                input_tensor_padded = np.pad(input_tensor, ((pad_y, pad_y), (pad_x, pad_x)), 'same',constant_values=0)
            output_y= (self.y+2*pad_y-self.m)/self.stride_shape[0]  +1
            output_x = (self.x+2*pad_x-self.n)/self.stride_shape[1] +1
            output_stack_num=self.channel_num
        else:
            pad_y = y-self.m
            if(pad_y!=0):
                input_tensor_padded = np.pad(input_tensor,pad_y,'same',constant_values=0)
                output_y = (self.y+2*pad_y-self.m)/self.stride_shape  +1
                output_stack_num=self.channel_num
