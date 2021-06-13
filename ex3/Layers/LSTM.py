from Layers import Base,TanH,Sigmoid,FullyConnected
import numpy as np
import copy

class LSTM(Base.BaseLayer):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.temp = np.zeros((4,hidden_size))
        self.hidden_state=np.zeros((hidden_size))
        self.c=np.zeros((hidden_size))

        self.memorize=False
        self.trainable = True
        self.act_hidden = np.empty([])
        np.append(self.act_hidden,Sigmoid.Sigmoid)
        np.append(self.act_hidden,Sigmoid.Sigmoid)
        np.append(self.act_hidden, TanH.TanH)
        np.append(self.act_hidden, Sigmoid.Sigmoid)
        self.ac_out=Sigmoid.Sigmoid
        self.fc_hidden=FullyConnected.FullyConnected(input_size+hidden_size,hidden_size)
        self.fc_output =FullyConnected.FullyConnected(hidden_size,output_size)

    @property
    def weights(self):
        stacked=self.fc_hidden.weights
        for i in range(3):
            stacked=np.concatenate([stacked,self.fc_hidden.weights],axis=1)
        return stacked
    @weights.setter
    def weights(self,w):
        self.fc_hidden.weights=w

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self,input_tensor):

        self.output=np.zeros((input_tensor.shape[0],self.output_size))
        for time in range(input_tensor.shape[0]):
            if (self.memorize==False and time==0):
                self.hidden_state=np.zeros((self.hidden_size))
                #add self.c initialization
                self.c=np.zeros((self.hidden_size))


            """
            concatenate input & hidden state multiply weights
            """
            con= np.concatenate([self.hidden_state,input_tensor[time]])
            con_f = self.fc_hidden.forward(con)
            con_i=con_c=con_o=con_f


            """
            apply 4 activation functions(get f,i,C,o)
            """
            # 1. get f
            sig_f = Sigmoid.Sigmoid()
            self.temp[0]=sig_f.forward(con_f)
            #2. get i
            sig_i =Sigmoid.Sigmoid()
            self.temp[1]=sig_i.forward(con_i)
            #get c
            tan_c =TanH.TanH()
            self.temp[2]=tan_c.forward(con_c)
            # get o
            sig_o = Sigmoid.Sigmoid()
            self.temp[3]=sig_o.forward(con_o)
            """
            do multiplication& addition(C)
            """
            i_c = np.dot(self.temp[1],self.temp[2])
            #c_f = np.matmul(np.asmatrix(self.c),np.asmatrix(self.temp[0]))
            c_f=np.dot(self.c,self.temp[0])
            self.c=i_c +c_f
            """
            apply tanh & multiplication(get hidden_state)
            """
            tan=TanH.TanH()
            ctan=tan.forward(self.c)
            self.hidden_state=self.temp[3]*ctan


            """
            apply sigmoid to get output tensor
            """
            sig=Sigmoid.Sigmoid()

            h=self.fc_output.forward(self.hidden_state)
            self.output[time]=sig.forward(h)




        return self.output
    def backward(self,error_tensor):
        prev_error=np.zeros((error_tensor.shape[0],self.input_size))
        """
        unsigmoid(get hidden state)
        """
        """
        backpropagate mul-> get gradient ctan &o
        backpropagate sigmoid with gradient o
        """
        """
        backpropagate summation of gradient of ctan->copy
        """
        """
        backpropagate multiplication->gradient of i &C
        backpropagate sigmoid & tanh
        
        """
        """
        backpropagate multiplication->get gradient X & f
        backpropagate sigmoid with gradient f
        """
        """
        split the output matrix to get gradient hiddent state&input
        """

        return prev_error