from Bayes import BayesLayer
import numpy as np
from Optimizers import Sgd as opt
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
        self.ouput_tensor=np.reshape(self.ouput_tensor,(self.batch_size,self.input_size))
        output=np.copy(self.ouput_tensor)
        return output
    def backward(self,error_tensor):
        self.prev_error = np.maximum(0,error_tensor)
        print(error_tensor)
        # compute gradient w.r.t weight
        self.input_tensor=np.reshape(self.input_tensor,(self.batch_size,self.input_size))
        #print(self.input_tensor)
        #print(error_tensor)
        #self.gradient_weights = np.matmul(self.input_tensor.T,error_tensor)

        self.weights= np.random.uniform(0,1,len(self.input_tensor)*len(self.input_tensor[0]))
        self.weights=np.reshape(self.weights,(len(self.input_tensor),len(self.input_tensor[0])))
        self.gradient_weights = np.matmul(self.input_tensor.T,error_tensor)
       # self.weights = self.calculate_update(self.weights, self.gradient_weights)

        return self.prev_error

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, learning_rate):
        self._optimizer = opt(learning_rate)

    def calculate_update(self,weight_tensor,gradient_tensor):
        #self._optimizer = opt(1)
        #self._optimizer(1)
        updated_weight = self._optimizer.calculate_update(weight_tensor,gradient_tensor)
        return updated_weight

