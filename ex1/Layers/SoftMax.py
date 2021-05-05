from Base import BayesLayer
import numpy as np
class SoftMax(BayesLayer):
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
        batch_sum=[]
        for i in range(len(error_tensor)):
            temp=0
            for j in range(len(error_tensor[i])):
                temp+= error_tensor[i][j]*self.output_tensor[i][j]
            batch_sum.append(temp)

        self.prev=[]
        for i in range(len(error_tensor)):
            error_tensor[i]=error_tensor[i]-batch_sum[i]

        for i in range(len(error_tensor)):
            temp=[]
            for j in range(len(error_tensor[0])):
                temp.append(self.output_tensor[i][j]*error_tensor[i][j])
            self.prev.append(temp)
        self.prev=np.reshape(self.prev,(len(error_tensor),len(error_tensor[0])))
        return self.prev
