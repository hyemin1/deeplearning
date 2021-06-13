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
        self.n=True
        #optimizer
        self._optimizer=None
        # initialize weight matrix
        self.weights = np.random.uniform(0, 1, ((self.input_size+1 ) * self.output_size))
        self.weights = np.reshape(self.weights, (self.input_size+1, self.output_size))
        self.par_weights = self.weights[0:-1,:]
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
    @gradient_weights.setter
    def gradient_weights(self,w):
        self.gradient_w=w

    @property
    def X(self):
        return self.input_tensor
    @X.setter
    def X(self,input):
        self.input_tensor=input

    def calculate_regularization_loss(self):

        return self._optimizer.regularizer.norm(self.weights)
    def forward(self,input_tensor):
        self.batch_size=len(input_tensor)
        #self.input_size=len(input_tensor[0])
        self.input_tensor=input_tensor

        if(len(self.input_tensor)>=2 ):
            #re-define input_tensor: add a col
            self.input_tensor = np.ones((self.batch_size,self.input_size+1))
            self.input_tensor[:,:-1]=input_tensor
            # compute output : Y = X*W
            self.output_tensor = np.matmul(self.input_tensor, self.weights)
        else:
            self.input_tensor=np.asmatrix(np.append(np.array(input_tensor),1))
           # self.input_tensor=input_tensor

            #compute output : Y = X*W
            self.output_tensor= np.array(np.matmul(np.asmatrix(self.input_tensor),self.weights))
            self.output_tensor=self.output_tensor[0]
        #print(self.output_tensor[0])
        #print(self.input_tensor.shape)
        return self.output_tensor

    def backward(self,error_tensor):
        #weights in this current layer
        self.curr_weight=self.weights

        #compute gradient w.r.t weight

        if((np.asmatrix(self.input_tensor).shape[0])>=2):
            self.gradient_w = np.matmul(self.input_tensor.T, error_tensor)
        else:
            self.gradient_w = np.matmul(np.asmatrix(self.input_tensor).T,np.asmatrix(error_tensor))

        #if the optimizer is set, update weights
        if (self._optimizer!=None):
            self.weights = self._optimizer.calculate_update(self.weights,self.gradient_weights)

        #delete biases of (unupdated) weights
        self.curr_weight = np.delete(self.curr_weight, len(self.weights) - 1, 0)

        # previous error tensor
        self.prev_error = np.matmul(error_tensor, self.curr_weight.T)

        return self.prev_error
