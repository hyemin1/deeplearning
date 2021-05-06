from Optimization import Optimizers
import copy
class NeuralNetwork():
    def __init__(self,sgd):
        self.optimizer=sgd
        self.loss=[]#contains loss values after calling train()
        self.layers=[] #contains all layers
        self.data_layer=None #contains input data & label
        self.loss_layer=None # contain cross entropy layer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        out=self.input_tensor
        for lay in self.layers:
            out=lay.forward(out)
        out=self.loss_layer.forward(out,self.label_tensor)

        return out
    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for lay in reversed(self.layers):
            error = lay.backward(error)

    def append_layer(self,layer):
        if layer.trainable==True:
            layer._optimizer =copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self,iterations):
        for i in range(iterations):
            out=self.forward()
            self.loss.append(out)
            self.backward()

    def test(self,input_tensor):
        out = input_tensor
        for lay in self.layers:
            out = lay.forward(out)
        return out
