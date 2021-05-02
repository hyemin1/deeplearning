from Base import BayesLayer
import numpy as np
class SoftMax(BayesLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        input_max = np.amax(input_tensor,axis=1,keepdims=True)
        reduced_exp = np.exp(input_tensor-input_max)
        reduced_sum = np.sum(reduced_exp,axis=1,keepdims=True)
        self.output_tensor = reduced_exp/reduced_sum
        return self.output_tensor

    def backward(self, error_tensor):
        scalar =((error_tensor.flatten())*(self.output_tensor.flatten())).sum()
        #print((error_tensor.flatten())*(self.output_tensor.flatten()))
        #print(type(error_tensor))
        self.prev=np.reshape((self.output_tensor.flatten())*(error_tensor.flatten()-scalar),(len(error_tensor),len(error_tensor[0])))
        # compute gradient w.r.t weight
        self.gradient_weights = np.matmul(self.input_tensor.T, error_tensor)
        return self.prev
