import numpy as np
from Layers import Base
class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations=None
        self.gradient_act=None
    def forward(self,input_tensor):
        self.activations=np.tanh(input_tensor)
        return self.activations
    def backward(self,error_tensor):
        self.gradient_act = (1-self.activations**2)*error_tensor
        return self.gradient_act