from Base import BayesLayer
import numpy as np

class SoftMax(BayesLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor):
        reduced_logit = input_tensor - np.amax(input_tensor)
        exponential_of_current_logit = np.exp(reduced_logit)
        exponential_of_all_logit = np.exp(input_tensor)
        sum_of_all_exponential_logit = np.sum(exponential_of_all_logit)
        self.output_tensor = exponential_of_current_logit/sum_of_all_exponential_logit
        return self.output_tensor

    def backward(self, error_tensor):
        sum_of_multiplication = np.sum(error_tensor * self.output_tensor)
        subtraction_from_current_tensor = error_tensor - sum_of_multiplication
        error_tensor_for_previous_layer = np.matmul(self.weights, subtraction_from_current_tensor)
        return error_tensor_for_previous_layer
