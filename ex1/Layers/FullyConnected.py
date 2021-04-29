from Bayes import BayesLayer
import numpy as np
class FullyConnected(BayesLayer):
    def __init__(self,input_size,output_size):
        BayesLayer.__init__(self)
        self.trainable=True
        self.input_size=input_size
        self.output_size=output_size
        self.weights= np.random.uniform(0,1,(self.output_size)*(self.input_size+1))
        self.weights=np.reshape((self.ouput_size,self.input_size+1))
        self.input_tensor=np.empty((1,1))
        self.batch_size=0
        self.gradient_weights=0
        
    #need to define optimizer

    def forward(self,input_tensor):
        self.batch_size=len(input_tensor)
        input_tensor=np.flatten(input_tensor)
        addition_ones  = np.ones((1,self.input_size))
        self.input_tensor=np.concatenate(input_tensor,addition_ones)
        self.input_tensor=np.reshape(self.input_tensor,(self.batch_size+1,self.input_size))
        self.output_tensor= np.matmul(self.weights,self.input_tensor)
        output=np.copy(self.output_tensor)
        return output
    def backward(self,error_tensor):
        self.prev_error = np.matmul(self.weights.T,error_tensor)
        prev = np.copy(self.prev_eror)
        #probably wrong
        self.gradient_weights = (np.matmul(self.weights*self.input_tensor,self.input_tensor.T))
        self.weights = self.calculate_update(self.weights,self.gradient_weights)
        return prev
    def calculate_update(weight_tensor,gradient_tensor):
        updated_weight = weight_tensor - np.matmul(gradient_tensor*(self.input_tensor.T))

        return updated_weight

