from Layers import Base
import numpy as np

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.batch_size=0
        self.input_size=0
    def forward(self,input_tensor):
        #re-assign batch_size and input_size
        self.batch_size = len(input_tensor)
        self.input_size = len(input_tensor[0])

        #make input_tensor 1D
        self.input_tensor = input_tensor.flatten()

        #apply ReLU function: max(0,x)
        self.ouput_tensor = np.maximum(0,self.input_tensor)

        #reshape
        self.ouput_tensor=np.reshape(self.ouput_tensor,(self.batch_size,self.input_size))
        self.input_tensor = np.reshape(self.input_tensor,(self.batch_size,self.input_size))

        output = np.copy(self.ouput_tensor)
        return output
    def backward(self,error_tensor):
        #get row, col for reshaping later
        row = len(error_tensor)
        col = len(error_tensor[0])

        self.input_tensor=self.input_tensor.flatten()
        #make error_tensor 1D
        error_tensor=error_tensor.flatten()
        #initialize prev_error with 0s
        self.prev_error=[0]*len(error_tensor)
        #apply ReLU function for backward computation
        #if x<=0:0, else:en
        for i in range(len(self.input_tensor)):
            if (self.input_tensor[i]>0):
                self.prev_error[i]=error_tensor[i]
        #reshape previous error tensor
        self.prev_error=np.reshape(self.prev_error,(row,col))

        return self.prev_error
