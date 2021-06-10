from Layers import Base
import numpy as np

class FullyConnected(Base.BaseLayer):
    def __init__(self,input_size,output_size):
        Base.BaseLayer.__init__(self)
        self.trainable=True #has to be trained
        self.input_size=input_size
        self.output_size=output_size
        self.input_tensor=np.empty((1,1))
        self.batch_size=0
        #optimizer
        self._optimizer=None
        # initialize weight matrix
        self.weights = np.random.uniform(0, 1, ((self.input_size+1 ) * self.output_size))
        self.weights = np.reshape(self.weights, (self.input_size+1, self.output_size))

    def initialize(self, weights_initializer, bias_initializer):
        # weights_initializer.fan_in = self.input_size
        # weights_initializer.fan_out = self.output_size
        self.weights[0:-1,:]=weights_initializer.initialize((self.input_size,self.output_size),self.input_size,self.output_size)
        self.weights[-1]=bias_initializer.initialize((1,self.output_size),self.input_size,self.output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self,sgd):
        self._optimizer = sgd

    @property
    def gradient_weights(self):
        return self.gradient_w

    @property
    def X(self):
        return self.input_tensor
    @X.setter
    def X(self,input):
        self.input_tensor=input
    @property
    def w(self):
        return self.weights
    @w.setter
    def w(self,w):
        self.weights=w
    def forward(self,input_tensor):
        self.batch_size=len(input_tensor)
        #self.input_size=len(input_tensor[0])
        self.input_tensor=input_tensor
        if(len(self.input_tensor)==2):
            #re-define input_tensor: add a col
            self.input_tensor = np.ones((self.batch_size,self.input_size+1))
            self.input_tensor[:,:-1]=input_tensor
        else:
            self.input_tensor=np.append(input_tensor,1)
           # self.input_tensor=input_tensor

        #compute output : Y = X*W
        self.output_tensor= np.array(np.matmul(np.asmatrix(self.input_tensor),self.weights))
        #print(self.output_tensor[0])

        return self.output_tensor[0]

    def backward(self,error_tensor):
        #weights in this current layer
        self.curr_weight=self.weights

        #compute gradient w.r.t weight
        self.gradient_w = np.matmul(np.asmatrix(self.input_tensor).T,np.asmatrix(error_tensor))

        #if the optimizer is set, update weights
        if (self._optimizer!=None):
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)

        #delete biases of (unupdated) weights
        self.curr_weight = np.delete(self.curr_weight, len(self.weights) - 1, 0)

        # previous error tensor
        self.prev_error = np.matmul(error_tensor, self.curr_weight.T)

        return self.prev_error
