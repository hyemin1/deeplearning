from Layers import Base
import numpy as np
class SoftMax(Base.BayesLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        #input_max = np.amax(input_tensor,axis=1,keepdims=True)
        input_max=np.amax(input_tensor)
        #print(input_max)
        reduced_exp = np.exp(input_tensor-input_max)
        reduced_sum = np.sum(reduced_exp,axis=1,keepdims=True)
        self.output_tensor = reduced_exp/reduced_sum

        return self.output_tensor
    
    def backward(self, error_tensor):
        temp_sum = error_tensor * self.output_tensor
        sum_by_rows = np.sum(temp_sum.T, axis=0, keepdims=True)
        batch_sum = sum_by_rows.flatten()

        tiled_batch_sum = np.tile(batch_sum, (len(error_tensor[0]),1)).T
        error_tensor = error_tensor - tiled_batch_sum

        prev_error_tensor = self.output_tensor * error_tensor
        return prev_error_tensor
