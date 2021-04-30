import numpy as np

class Sgd():
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate

    def calculate_update(self,weight_tensor,gradient_tensor):
        #updated weight = weight - learningrate * gradient(weight)

        #np.gradient(gradient_tensor,weight_tensor)
        print(self.learning_rate)
        print(gradient_tensor)
        self.updated_weight = weight_tensor - (self.learning_rate)*(gradient_tensor)

        return self.updated_weight
