from Loss import CrossEntropyLoss
from FullyConnected import FullyConnected
import copy

class NeuralNetwork(CrossEntropyLoss, FullyConnected):
    def __init__(self):
        self.optimizer = None
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

        self.input_tensor = None
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        last_layer_output = self.forward(self.input_tensor, self.label_tensor)
        return last_layer_output

    def backward(self):
        self.loss_layer = self.backward(self.label_tensor)
        return self.loss

    def append_layer(self, layer):
        fully_connected = FullyConnected()
        if (self.trainable):
            self.optimizer = copy.deepcopy(fully_connected.optimizer)
            self.layers.append(layer)

    def train(self, iterations):


    def test(self, input_tensor):
