import numpy as np
class CrossEntropyLoss():
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor,label_tensor):
        #store to use at backward method
        self.prob_label =input_tensor

        #compute loss
        loss = self.prob_label[np.arange(len(self.prob_label)),np.argmax(label_tensor,axis=1)]
        loss += np.finfo(float).eps
        self.loss = (-(np.log(loss))).sum()

        return self.loss
    def backward(self,label_tensor):
        #create a error tensor with same shape as given label tensor
        self.prev_error = label_tensor
        #assign values with given equation
        self.prev_error=(-1)*(np.divide(label_tensor,self.prob_label))

        return self.prev_error
