import numpy as np
class BaseLayer():
    def __init__(self):
        self.trainable=False
        #self.weights=np.empty((1,1))
        self.input_size=0
        self.output_size=0
        self.testing_phase=False