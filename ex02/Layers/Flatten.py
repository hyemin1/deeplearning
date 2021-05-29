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
        self.batch=len(input_tensor)
        self.ch = len(input_tensor[0])
        self.height = len(input_tensor[0][0])
        if(self.height!=0):
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
        if(self.width!=0):
            self.error_unflatt = np.reshape(error_tensor,(self.batch,self.ch,self.height,self.width))
        #to 3D
        else:
            self.error_unflatt=np.reshape(error_tensor,(self.batch,self.ch,self.height))
        return self.error_unflatt
