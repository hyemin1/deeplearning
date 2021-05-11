import numpy as np
import math
class Sgd():
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate


    def calculate_update(self,weight_tensor,gradient_tensor):
        #update weights: W - (learning rate)*X^T*E
        self.updated_weight = weight_tensor - (self.learning_rate)*(gradient_tensor)
        return self.updated_weight



class SgdWithMomentum:
    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate= learning_rate
        self.momentum_rate=momentum_rate
        self.velocity=0
    def calculate_update(self,weight_tensor,gradient_tensor):
        self.velocity = self.momentum_rate*self.velocity-self.learning_rate*gradient_tensor
        updated_weight = weight_tensor+self.velocity
        return updated_weight



class Adam:
    def __init__(self,learning_rate,mu,rho):
        self.learning_rate = learning_rate
        self.mu=mu
        self.rho=rho
        self.velocity=0
        self.r=0
        self.t=0
    def calculate_update(self,weight_tensor,gradient_tensor):
        self.t=self.t+1
        self.g = gradient_tensor
        self.velocity=self.mu*self.velocity+(1-self.mu)*self.g
        self.velocityb = self.velocity/(1-(self.mu**self.t))
        self.r = self.rho*self.r + np.multiply((1-self.rho)*self.g,self.g)
        self.rb = self.r/(1-(self.rho**self.t))
        updated_weight = weight_tensor - self.learning_rate*(self.velocityb/(math.sqrt(self.rb)+np.finfo(float).eps))
        return updated_weight
