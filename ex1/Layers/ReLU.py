from Bayes import BayesLayer
import numpy as np
class ReLU(BayesLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        row = len(input_tensor)
        col = len(input_tensor[0])
        self.input_tensor = input_tensor.flatten()
        self.ouput_tensor = np.array(max(0,x) for x in self.input_tensor)
        self.ouput_tensor=np.reshape((row,col))
        output=np.copy(self.ouput_tensor)
        return output
    def backward(self,error_tensor):
        return
