from Bayes import BayesLayer
from Optimizers import Sgd as opt
import numpy as np
class FullyConnected(BayesLayer):
    def __init__(self,input_size,output_size):
        BayesLayer.__init__(self)
        self.trainable=True
        self.input_size=input_size
        self.output_size=output_size
        #self.weights= np.random.uniform(0,1,(self.input_size+1)*(self.output_size))
        #self.weights=np.reshape(self.weights,(self.input_size+1,self.output_size))
        self.input_tensor=np.empty((1,1))
        self.batch_size=0
        #self.gradient_weights=[0]

        #optimizer
        self._optimizer=None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,sgd):
        self._optimizer = sgd
        print("here")
        #print(learning_rate)
    @property
    def gradient_weights(self):
        return self.gradient_w
    def forward(self,input_tensor):
        self.batch_size=len(input_tensor)

        #re-define input_tensor: add a col
        self.input_tensor = np.ones((self.batch_size,self.input_size+1))
        self.input_tensor[:,:-1]=input_tensor
        #print(len(self.input_tensor))
        #print(self.input_tensor)
        #re-define weight
        self.weights = np.random.uniform(0,1,((self.input_size+1)*self.output_size))
        self.weights = np.reshape(self.weights,(self.input_size+1,self.output_size))

        self.output_tensor= np.matmul(self.input_tensor,self.weights)
        output=np.copy(self.output_tensor)
        return output
    #@property
    #def gradient_weights(self):
    #    return self.gradient_weights
    def backward(self,error_tensor):
        #remove the column with 1s of weights
        #self.weights=np.delete(self.weights,-1,1)
        #Error tensor from the previous layer
        # delete the last row(bias) of weights
        self.weights = np.delete(self.weights, len(self.weights) - 1, 0)
        #previous error tensor
        self.prev_error = np.matmul(error_tensor,self.weights.T)
        prev = np.copy(self.prev_error)

        self.curr_erro=error_tensor
        #remove the col with 1s(bias) of input tensor
        self.input_tensor = np.delete(self.input_tensor,len(self.input_tensor[0])-1,1)
        print(self.input_tensor.shape)
        print(self.weights.shape)
        print(error_tensor.shape)
        #compute gradient w.r.t weight
        self.gradient_w = np.matmul(self.input_tensor.T,error_tensor)
        #self.gradient_weights = self.calculate_update(self.weights, np.matmul(self.input_tensor.T,error_tensor))
        #self.weights = self.calculate_update(self.weights,self.gradient_weights)
        self.gradient_weights
        if (self._optimizer!=None):
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)
        return prev

    #def calculate_update(self,weight_tensor,gradient_tensor):
        #self._optimizer = opt(1)
        #self._optimizer(1)
        #updated_weight = self._optimizer.calculate_update(weight_tensor,gradient_tensor)
        #return updated_weight
