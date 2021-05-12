from Layers import Base
import numpy as np
class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.a=len(input_tensor)
        self.b = len(input_tensor[0])
        self.c = len(input_tensor[0][0])
        self.d=len(input_tensor[0][0][0])
        self.input_flatt = []
        for i in range(len(input_tensor)):
            temp=input_tensor[i].flatten()
            self.input_flatt.append(temp)
        return self.input_flatt
    def backward(self,error_tensor):
        self.error_flatt = np.reshape(error_tensor,(self.a,self.b,self.c,self.d))
        return self.error_flatt
