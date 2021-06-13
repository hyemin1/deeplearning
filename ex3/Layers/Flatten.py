from Layers import Base
import numpy as np
class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.batch=0
        self.ch=0
        self.height=0
        self.width=0

    def forward(self,input_tensor):
        #store the shape of input tensor
        self.input_tensor = input_tensor
        self.batch=len(input_tensor)
        self.ch = len(input_tensor[0])
        if len(input_tensor.shape) > 2:
            self.height = len(input_tensor[0][0])

        if len(input_tensor.shape) > 3:
         self.width=len(input_tensor[0][0][0])

        self.input_flatt = []
        #flat matrix for all batches
        for i in range(self.batch):
            temp=input_tensor[i].flatten()
            self.input_flatt.append(temp)
        return self.input_flatt
    def backward(self,error_tensor):
        #reshape error tensor
        #to 4D
        if(len(self.input_tensor.shape) == 4):
            self.error_unflatt = np.reshape(error_tensor,(self.batch,self.ch,self.height,self.width))
        #to 3D
        elif (len(self.input_tensor.shape) == 3):
            self.error_unflatt=np.reshape(error_tensor,(self.batch,self.ch,self.height))
        else:
            self.error_unflatt = np.reshape(error_tensor, (self.batch, self.ch))
        return self.error_unflatt