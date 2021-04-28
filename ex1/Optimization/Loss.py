from Bayes import BayesLayer
import numpy as np
class CrossEntropyLoss(BayesLayer):
    def __init__(self):
        super().__init__()

    def forward(self,input_tensor,label_tensor):
        self.loss = np.sum(-(np.log(yk+np.finfo.eps) for yk in label_tensor if yk==1))
        loss=np.copy(self.loss)
        return loss
