from Layers import Base
import numpy as np
class Pooling(Base.BaseLayer):
    def __init__(self,stride_shape,pooling_shape):
        super().__init__()
        self.stride_shape =stride_shape
        self.pooling_shape=pooling_shape
        self.img_2d=True
        #store shape of stride
        self.s_height =self.stride_shape[0]
        self.s_width =self.stride_shape[1]
        #store shape of kernel
        self.k_height =self.pooling_shape[0]
        self.k_width =self.pooling_shape[1]

    def forward(self,input_tensor):
        #store input tensor to use the shape on backward
        self.input_tensor=input_tensor
        #number of batches in input tensor
        self.b=input_tensor.shape[0]
        #number of input channels
        self.input_channel =input_tensor.shape[1]
        #height
        self.height = input_tensor.shape[2]
        #width
        self.width = input_tensor.shape[3]

        #calculate output shape
        out_height = int((self.height - self.k_height) / self.s_height) + 1
        out_width = int((self.width - self.k_width) / self.s_width) + 1

        #create output
        self.output = np.zeros((self.b,self.input_channel,out_height,out_width))
        #store indices of max values

        #for every batch
        for batch in range(self.b):
            #for all input channels per batch
            for channel in range(self.input_channel):
                # apply max pooling for all values of 2D matrix
                for r in range(out_height):
                    #set the starting and endpoint of row
                    row_kerstart = r * self.s_height
                    row_kerend = row_kerstart + self.k_height

                    for c in range(out_width):
                        col_kerstart = c * self.s_width
                        col_kerend = col_kerstart + self.k_width

                        #sotre the vale of max value of sub-region(calculated using starting points & polling matrix&stride)
                        self.output[batch, channel, r, c] = np.max(input_tensor[batch, channel, row_kerstart:row_kerend, col_kerstart:col_kerend])

        return self.output

    def backward(self,error_tensor):

        #calculate the height& width of upsampled matrix
        # upsample_height = int((self.height - self.k_height) / self.s_height) + 1
        # upsample_width = int((self.width - self.k_width) / self.s_width) + 1


        #create up-sampled tensor
        upsampled = np.zeros((self.b,self.input_channel,self.height,self.width))
        # for every batch
        for batch in range(self.b):
            #for all channels per batch
            for channel in range(self.input_channel):
                #for all 2D matrix
                for r in range(error_tensor.shape[2]):
                    # set the starting and endpoint of row
                    row_ker_start = r * self.s_height
                    row_ker_end = row_ker_start + self.k_height
                    for c in range(error_tensor.shape[3]):
                        col_ker_start = c * self.s_width
                        col_ker_end = col_ker_start + self.k_width

                        #get the sub-region of input tensor using pooling shape and starting point
                        subset = self.input_tensor[batch, channel, row_ker_start:row_ker_end, col_ker_start:col_ker_end]
                        #get the max. value of that sub-region
                        #if the pixel value is equal to the max. value, add the value of error tensor
                        #if not, 0 will be added
                        #as same pixels can be included more than 1 times, use addition
                        upsampled[batch, channel, row_ker_start:row_ker_end, col_ker_start:col_ker_end] += (subset == np.max(subset)) * error_tensor[batch, channel, r, c]

        return upsampled

