from Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        
    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        input_tensor[input_tensor < 0] = 0
        return input_tensor

    def backward(self,error_tensor):
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor
