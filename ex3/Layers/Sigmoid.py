from Layers import Base
import numpy as np

class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.activations=None
    def forward(self,input_tensor):
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self,error_tensor):
        self.gradient=np.multiply(np.multiply(self.activations,(1-self.activations)),error_tensor)

        return self.gradient