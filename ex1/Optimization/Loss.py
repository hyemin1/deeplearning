
from SoftMax import SoftMax as sft
import numpy as np
class CrossEntropyLoss():
    def __init__(self):
        super().__init__()
        self.input_size=0
        self.soft =sft()
    def forward(self,input_tensor,label_tensor):
        self.prob_label =input_tensor

        #compute loss
        loss = self.prob_label[np.arange(len(self.prob_label)),np.argmax(label_tensor,axis=1)]
        loss += np.finfo(float).eps
        self.loss = (-(np.log(loss))).sum()

        return self.loss
    def backward(self,label_tensor):
        self.prev_error = -(np.divide(label_tensor,self.prob_label))
        return self.prev_error
