import copy
import numpy as np

class NeuralNetwork():
    def __init__(self, sgd, weights_initializer, bias_initializer):
        self.optimizer = sgd
        self.loss = []#contains loss values after calling train()
        self.layers = [] #contains all layers
        self.data_layer = None #contains input data & label
        self.loss_layer = None # contain cross entropy layer
        self.bias_initializer = bias_initializer
        self.weights_initializer=weights_initializer

    @property
    def phase(self):
        pass
    @phase.setter
    def phase(self,p):
        for lay in self.layers:
            lay.testing_phase=p
    # def norm(self,weights):
    #     total=0
    #     # for lay in self.layers:
    #     #     if (lay.trainable==True):
    #     #             if(lay.optimizer.regularizer!=None and lay.n==True):
    #     #
    #     #                 total+=lay.calculate_regularization_loss()
    #     #             #print(lay.calculate_regularization_loss())
    #
    #     return total
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        out = self.input_tensor
        for lay in self.layers:
            out = lay.forward(out)
        out = self.loss_layer.forward(out, self.label_tensor)

        return out
    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for lay in reversed(self.layers):
            error = lay.backward(error)

    def append_layer(self,layer):
        if layer.trainable==True:
            layer.optimizer = copy.deepcopy(self.optimizer)
            #layer.weights = initialized_weights
            layer.initialize(self.weights_initializer,self.bias_initializer)
            #layer.initialize()

        self.layers.append(layer)

    def train(self,iterations):
        self.phase=False
        for i in range(iterations):
            out=self.forward()
            self.loss.append(out)
            #self.total=0
            # for lay in self.layers:
            #
            #     if(lay.trainable == True):
            #         if (lay.optimizer.regularizer != None and lay.n == True):
            #             pass
                        #self.total+=lay.calculate_regularization_loss()
                        #print(lay.calculate_regularization_loss())


            #self.loss[-1]=self.loss[-1]+total
            #self.loss=np.add(self.loss,self.total)
            #np.add(self.loss,l)
            #self.loss=np.add(self.loss, self.norm(0))
            self.backward()


    def test(self,input_tensor):
        self.phase=True
        out = input_tensor
        for lay in self.layers:
            out = lay.forward(out)
        return out
