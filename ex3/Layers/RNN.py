from Layers import Base,TanH,Sigmoid,FullyConnected
import numpy as np
import copy
class RNN(Base.BaseLayer):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size = hidden_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable=True
        self.memorize=False
        self.fc_output=FullyConnected.FullyConnected(self.hidden_size,self.output_size)
        self.fc_hidden = FullyConnected.FullyConnected(self.input_size+self.hidden_size,self.hidden_size)
        self.hidden_opt=None
        self.out_opt=None
        self.reg=True

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer,bias_initializer)
        self.fc_output.initialize(weights_initializer,bias_initializer)

    @property
    def weights(self):
        return self.fc_hidden.weights
    @weights.setter
    def weights(self,nw):
        self.fc_hidden.weights=nw
    @property
    def gradient_weights(self):
        return self.gradient_w_hidden
    @gradient_weights.setter
    def gradient_weights(self,w):
        self.fc_hidden.gradient_weights(w)

    @property
    def optimizer(self):
        return self.hidden_opt
    @optimizer.setter
    def optimizer(self,opt):
        self.hidden_opt = copy.deepcopy(opt)
        self.out_opt = copy.deepcopy(opt)


    def calculate_regularization_loss(self):
        return self.hidden_opt.regularizer.norm(self.fc_hidden.weights)
    def forward(self,input_tensor):

        output_tensor=np.zeros((input_tensor.shape[0],self.output_size))
        self.input_tensor = input_tensor

        self.sigmoid_values=np.empty((input_tensor.shape[0]),dtype=Sigmoid.Sigmoid)
        self.tanh_values=np.empty((input_tensor.shape[0]),dtype=TanH.TanH)

        self.hidden_values=np.zeros((input_tensor.shape[0],self.hidden_size))


        self.input_hidden=np.zeros((input_tensor.shape[0],self.input_size+self.hidden_size+1))
        self.input_out = np.zeros((input_tensor.shape[0], self.hidden_size+1))

        first=True
        for time in range(input_tensor.shape[0]):
            if (first==True):
                if(self.memorize==False):
                    self.hidden_state= np.zeros((self.hidden_size))

                first=False
            """
            concatenate input tensor
            """
            concatenated = np.concatenate([input_tensor[time], self.hidden_state])

            """
            apply hidden weight matrix
            """
            concatenated=self.fc_hidden.forward(np.asmatrix(concatenated))
            self.input_hidden[time]=self.fc_hidden.input_tensor
            """
            tanh,sigh for this iteration
            """
            tan = TanH.TanH()
            sig = Sigmoid.Sigmoid()
            """
            apply tanh activation function
            """
            self.hidden_state =tan.forward(concatenated)
            self.tanh_values[time] = tan
            self.hidden_values[time]=self.hidden_state

            """
            apply weight matrix for output
            """
            con_hidden=self.hidden_state
            con_hidden=self.fc_output.forward(np.asmatrix(con_hidden))
            self.input_out[time]=self.fc_output.input_tensor
            """
            apply sigmoid activation function & store
            """
            output_tensor[time]=sig.forward(con_hidden)
            self.sigmoid_values[time]=sig
        return output_tensor
    def backward(self,error_tensor):
        prev_error= np.zeros((error_tensor.shape[0],self.input_size))
        self.gradient_w_out = np.zeros((self.fc_output.weights.shape))
        self.gradient_w_hidden = np.zeros((self.input_size + self.hidden_size + 1, self.hidden_size))

        for time in reversed(range(error_tensor.shape[0])):
            """
            backpropagate sigmoid function
            """
            un_sig = self.sigmoid_values[time].backward(error_tensor[time])

            """
            backpropagate the output fully connected weight matrix
            """
            self.fc_output.input_tensor=self.input_out[time]
            un_out_weight=self.fc_output.backward(un_sig)
            """
            sum up the gradient w.r.t. output weights
            """
            self.gradient_w_out+=self.fc_output.gradient_weights
            """
            backpropagate copy operation
            """
            if(time==error_tensor.shape[0]-1):
                self.gradient_hidden=np.zeros((self.hidden_size))
            un_copy= un_out_weight+self.gradient_hidden
            """
            backpropagate tanh function
            """
            un_tanh =self.tanh_values[time].backward(un_copy)
            """
            backpropagate the hidden fully connected weight matrix
            """
            self.fc_hidden.input_tensor=(self.input_hidden[time])
            un_hidden_weight=self.fc_hidden.backward(un_tanh)
            """
            split the un_hidden_weight
            get the gradient w.r.t. input &hidden state
            """
            prev_error[time]=un_hidden_weight[0:self.input_size]
            self.gradient_hidden=un_hidden_weight[self.input_size:]
            """
            sum up gradient w.r.t. hidden weight matrix
            """
            self.gradient_w_hidden+=self.fc_hidden.gradient_weights
        """
        update fully connected hidden & output weight matrices
        """
        if(self.hidden_opt!=None):
            self.fc_hidden.weights=self.hidden_opt.calculate_update(self.fc_hidden.weights,self.gradient_w_hidden)
        if (self.out_opt != None):
            self.fc_output.weights = self.out_opt.calculate_update(self.fc_output.weights, self.gradient_w_out)

        return prev_error
