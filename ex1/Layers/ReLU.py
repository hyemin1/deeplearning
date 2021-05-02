from Bayes import BayesLayer
import numpy as np

class ReLU(BayesLayer):
    def __init__(self):
        super().__init__()
        self.batch_size=0
        self.input_size=0
    def forward(self,input_tensor):
        #re-assign batch_size and input_size
        self.batch_size = len(input_tensor)
        self.input_size = len(input_tensor[0])

        #make input_tensor 1D
        self.input_tensor = input_tensor.flatten()

        #apply ReLU function
        self.ouput_tensor = np.maximum(0,self.input_tensor)

        #reshape
        print(self.ouput_tensor.shape)
        self.ouput_tensor=np.reshape(self.ouput_tensor,(self.batch_size,self.input_size))
        self.input_tensor = np.reshape(self.input_tensor,(self.batch_size,self.input_size))
        output=np.copy(self.ouput_tensor)
        return output
    def backward(self,error_tensor):
        #get row, col for reshaping later
        row = len(error_tensor)
        col = len(error_tensor[0])

        self.input_tensor=self.input_tensor.flatten()
        #make error_tensor 1D
        error_tensor=error_tensor.flatten()
        #initialize prev_error with 0s
        self.prev_error=[0]*len(error_tensor)
        #apply ReLU function for backward computation
        for i in range(len(self.input_tensor)):
            if (self.input_tensor[i]>0):
                self.prev_error[i]=error_tensor[i]
        #reshape previous error tensor
        self.prev_error=np.reshape(self.prev_error,(row,col))
        #reshape current error tensor
        error_tensor = np.reshape(error_tensor,(row,col))
        #reshape input tensor
        self.input_tensor=np.reshape(self.input_tensor,(self.batch_size,self.input_size))

        #compute gradient w.r.t weights
        self.gradient_weights = np.matmul(self.input_tensor.T,error_tensor)

        return self.prev_error
