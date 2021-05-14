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

        #number of batches we have
        self.batch_num=self.convolution_shape[0]
        #number of channels per batch
        self.channel_num=self.convolution_shape[1]

        self.y=self.convolution_shape[2]
        #in case of 2D input
        if (len(convolution_shape)==4):
            self.x=self.convolution_shape[3]
            #initialize weights using given kernel shape
            self.weights=np.randpm.uniform(0,1,(self.channel_num*self.y*self.x))
            self.weights=np.reshape(self.weights,(self.channel_num,self.y,self.x))
        #in case of 1D input
        else:
            # initialize weights using given kernel shape
            self.weights=np.random.uniform(0,1,(self.channel_num*self.y))
            self.weights=np.reshape(self.weights,(self.channel_num,self.y))
        self.bias=None

    @property
    def gradient_weights(self):
        return self.gradient_w
    @property
    def gradient_bias(self):
        return self.gradient_b
    def forward(self,input_tensor):

        #also based on the assumption 2D. should be corrected to deal with 1D
        #zero padding to get output matrix after convolution which has same size of input_tensor
        pad_x = input_tensor.shape[1]-self.convolution_shape.shape[1]
        pad_y = input_tensor.shape[2]-self.convolution_shape.shape[2]
        if(pad_x!=0 or pad_y!=0):
            input_tensor_padded = np.pad(input_tensor,((pad_x,pad_x),(pad_y,pad_y)),'constant',constant_values=0)

        #calculate output height & width
        #also based on 2D input matrix
        #based on scalar stride shape
        #must be fixed

        output_height = ((input_tensor.shape[1] + 2 * pad_y-self.convolution_shape.shape[2])/self.stride_shape)+1
        output_width=((input_tensor.shape[2]+2*pad_x-self.convolution_shape[2])/self.stride_shape)+1
