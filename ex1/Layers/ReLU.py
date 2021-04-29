from Bayes import BayesLayer
import numpy as np
class ReLU(BayesLayer):
    def __init__(self):
        super().__init__()
        self.batch_size=0
        self.input_size=0
    def forward(self,input_tensor):
        self.batch_size = len(input_tensor)
        self.input_size = len(input_tensor[0])
        self.input_tensor = input_tensor.flatten()

        self.ouput_tensor = np.maximum(0,self.input_tensor)
        self.ouput_tensor=np.reshape((self.batch_size,self.input_size))
        output=np.copy(self.ouput_tensor)
        return output
    def backward(self,error_tensor):
        return
