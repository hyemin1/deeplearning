from Bayes import BayesLayer
import numpy as np
class SoftMax(BayesLayer):
    def __init__(self):
        super().__init__()
    def forward(self,input_tensor):
        self.output_tresor = np.exp(input_tensor-max(input_tensor))/(np.sum(np.exp(input_tensor)))