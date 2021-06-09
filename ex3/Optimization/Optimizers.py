import numpy as np
import math
class Optimizer:
    def __init__(self):
        self.regularizer=None

    def add_regularizer(self,regularizer):
        self.regularizer=regularizer

class Sgd(Optimizer):
    def __init__(self,learning_rate):
        self.learning_rate=learning_rate
        self.regularizer = None

    def calculate_update(self,weight_tensor,gradient_tensor):
        if(self.regularizer != None):
            self.updated_weight=weight_tensor-self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)-self.learning_rate*gradient_tensor
        else:
            self.updated_weight = weight_tensor - (self.learning_rate)*(gradient_tensor)
        return self.updated_weight


class SgdWithMomentum(Optimizer):
    def __init__(self,learning_rate,momentum_rate):
        self.learning_rate= learning_rate
        self.momentum_rate=momentum_rate
        self.velocity=0
        self.regularizer = None

    def calculate_update(self,weight_tensor,gradient_tensor):
        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        if (self.regularizer != None):
            self.updated_weight = weight_tensor - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)+self.velocity
        else:
            self.updated_weight = weight_tensor+self.velocity
        return self.updated_weight


class Adam(Optimizer):
    def __init__(self,learning_rate,mu,rho):
        self.learning_rate = learning_rate
        self.mu=mu
        self.rho=rho
        self.velocity=0
        self.r=0
        self.t=0
        self.regularizer = None

    def calculate_update(self,weight_tensor,gradient_tensor):
        self.t=self.t+1
        self.g = gradient_tensor
        self.velocity=self.mu*self.velocity+np.multiply((1-self.mu),self.g)
        self.velocityb = self.velocity/(1-(self.mu**self.t))
        self.r = self.rho*self.r + (1-self.rho)*self.g**2
        self.rb = self.r/(1-(self.rho**self.t))

        if(self.regularizer!=None):
            updated_weight = np.subtract(weight_tensor, np.multiply(self.learning_rate, (
                        self.velocityb / (np.sqrt(self.rb) + np.finfo(float).eps)))) - self.learning_rate*self.regularizer.calculate_gradient(weight_tensor)
        else:
            updated_weight = np.subtract(weight_tensor,np.multiply(self.learning_rate,(self.velocityb/(np.sqrt(self.rb)+np.finfo(float).eps))))
        return updated_weight
