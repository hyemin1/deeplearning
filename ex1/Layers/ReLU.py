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
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor
