import numpy as np
from Layers import Base
class Dropout(Base.BaseLayer):
    def __init__(self,probability):
        super().__init__()
        self.p=probability
    def forward(self,input_tensor):
        if(self.testing_phase==False):
            self.dropped=input_tensor*(np.random.random(input_tensor.shape)>(1-self.p))
            return (1/self.p)*self.dropped
        else:
            return input_tensor

    def backward(self,error_tensor):
        return self.dropped*error_tensor*(1/self.p)
