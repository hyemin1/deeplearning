from Base import BayesLayer
import numpy as np
class FullyConnected(BayesLayer):
    def __init__(self,input_size,output_size):
        BayesLayer.__init__(self)
        self.trainable=True
        self.input_size=input_size
        self.output_size=output_size
        self.input_tensor=np.empty((1,1))
        self.batch_size=0
        #optimizer
        self._optimizer=None
        # define weight matrix
        self.weights = np.random.uniform(0, 1, ((self.input_size + 1) * self.output_size))
        self.weights = np.reshape(self.weights, (self.input_size + 1, self.output_size))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,sgd):
        self._optimizer = sgd
    @property
    def gradient_weights(self):
        return self.gradient_w
    def forward(self,input_tensor):
        self.batch_size=len(input_tensor)

        #re-define input_tensor: add a col
        self.input_tensor = np.ones((self.batch_size,self.input_size+1))
        self.input_tensor[:,:-1]=input_tensor
        #compute output : Y = X*W
        self.output_tensor= np.matmul(self.input_tensor,self.weights)

        return self.output_tensor

    def backward(self,error_tensor):
        #weights in this current layer
        self.curr_weight=self.weights

        #compute gradient w.r.t weight
        self.gradient_w = np.matmul(self.input_tensor.T,error_tensor)

        #self.gradient_weights
        #if the optimizer is set, update weights
        if (self._optimizer!=None):
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)
        #delete biases of (unupdated) weights
        self.curr_weight = np.delete(self.curr_weight, len(self.weights) - 1, 0)

        # previous error tensor
        self.prev_error = np.matmul(error_tensor, self.curr_weight.T)

        return self.prev_error
