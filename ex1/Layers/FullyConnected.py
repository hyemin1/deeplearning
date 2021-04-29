from Bayes import BayesLayer
from Optimizers import Sgd
import numpy as np
class FullyConnected(BayesLayer):
    def __init__(self,input_size,output_size):
        BayesLayer.__init__(self)
        self.trainable=True
        self.input_size=input_size
        self.output_size=output_size
        self.weights= np.random.uniform(0,1,(self.output_size)*(self.input_size+1))
        self.weights=np.reshape((self.ouput_size,self.input_size))
        self.input_tensor=np.empty((1,1))
        self.batch_size=0
        self.gradient_weights=0

        #optimizer
        self._optimizer=Sgd(1)

    @property
    def optimizer(self):
        return self._optimizer

    @property.setter
    def optimizer(self,learning_rate):
        self._optimizer = Sgd(learning_rate)

    def forward(self,input_tensor):
        self.batch_size=len(input_tensor)

        #re-define input_tensor
        input_tensor=np.flatten(input_tensor)
        addition_ones  = np.ones((1,self.input_size))
        self.input_tensor=np.concatenate(input_tensor,addition_ones)
        self.input_tensor = np.reshape(self.input_tensor, (self.batch_size + 1, self.input_size))

        #re-define weights
        addition_ones = np.ones((self.batch_size,1))
        np.concetenate((self.weights,addition_ones),axis=1)


        self.output_tensor= np.matmul(self.weights,self.input_tensor)
        output=np.copy(self.output_tensor)
        return output
    def backward(self,error_tensor):
        #remove the column with 1s of weights
        self.weights=np.delete(self.weights,-1,1)
        #Error tensor from the previous layer
        self.prev_error = np.matmul(self.weights.T,error_tensor)
        prev = np.copy(self.prev_eror)

        #remove the row with 1s of input tensor
        self.input_tensor = np.delete(self.input_tensor,len(self.input_tensor)-1,0)

        #compute gradient w.r.t weight
        self.gradient_weights = np.matmul(self.input_tensor,self.prev_error)

        self.weights = self.calculate_update(self.weights,self.gradient_weights)
        return prev

    def calculate_update(self,weight_tensor,gradient_tensor):
        return self._optimizer.calculate_update(weight_tensor,gradient_tensor)

