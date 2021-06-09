from Layers import Base,TanH,Sigmoid
import numpy as np
class RNN(Base.BaseLayer):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size = hidden_size
        self.hidden_state = np.zeros(hidden_size)
        self.trainable=True
        self.memorize=False
        self.w=np.random.uniform(0,1,(self.input_size+hidden_size+1, self.hidden_size))
        self.weights_hy = np.random.uniform(0,1,(hidden_size,output_size))
        self.bias = np.random.uniform(0,1,output_size)

        self.gradient_w=None
        self.gradient_b=None
        self.hidden_values=None
        self.gradient_hidden=np.zeros(hidden_size)

    def initialize(self, weights_initializer, bias_initializer):
        # weights_initializer.fan_in = self.input_size
        # weights_initializer.fan_out = self.output_size
        self.w[0:-1, :] = weights_initializer.initialize((self.weights.shape[0]-1,self.weights.shape[1]), self.input_size,
                                                               self.output_size)
        self.w[-1] = bias_initializer.initialize((1, self.weights.shape[1]), self.input_size, self.output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def weights(self):
        return self.w
    @weights.setter
    def weights(self,w):
        self.w = w
    @property
    def gradient_weights(self):
        return self.gradient_w
    @property
    def gradient_bias(self):
        return self.gradient_b
    def forward(self,input_tensor):
        output_tensor=np.zeros((input_tensor.shape[0],self.output_size))
        self.input_tensor = input_tensor
        self.sigmoid_values=np.empty((input_tensor.shape[0]),dtype=Sigmoid.Sigmoid)
        self.tanh_values=np.empty((input_tensor.shape[0]),dtype=TanH.TanH)
        self.hidden_values=np.zeros((input_tensor.shape[0],self.hidden_size))
        self.concatenated =np.zeros((input_tensor.shape[0],self.input_size+self.hidden_size+1))

        #self.input_tensor=np.zeros((input_tensor.shape[0],self.input_size+self.hidden_size+1))
        first=True
        for time in range(input_tensor.shape[0]):
            if (first==True):
                if(self.memorize==False):
                    self.hidden_state= np.zeros((self.hidden_size))

                first=False

            concatenated = np.concatenate([input_tensor[time], self.hidden_state])
            concatenated = np.append(concatenated, 1)
            self.concatenated[time]=concatenated
                #self.input_tensor[time]=concatenated
            concatenated=np.matmul(concatenated,self.weights)
            tan = TanH.TanH()
            sig = Sigmoid.Sigmoid()
               # self.sigmoid_values[time]=sig
                #np.append(self.sigmoid_values,sig)


            self.hidden_state = tan.forward(concatenated)
            self.tanh_values[time] = tan
                #self.tanh_values[time]=tan
                #np.append(self.tanh_values,tan)
            self.hidden_values[time]=self.hidden_state
            output_tensor[time]=sig.forward(np.matmul(self.hidden_state,self.weights_hy)+self.bias)
            self.sigmoid_values[time] = sig
                #self.sigmoid_values[time]=sig



        return output_tensor
    def backward(self,error_tensor):
        prev_error= np.zeros((error_tensor.shape[0],self.input_size))
        self.gradient_b=np.zeros((self.output_size))
        self.gradient_w=np.zeros((self.w.shape))
        self.gradient_why=np.zeros(self.weights_hy.shape)
        #gradient_b_tmp = np.zeros((error_tensor.shape[0]))
        """
        1. gradient w.r.t. input tensor
        2. gradient w.r.t. weights
        3. gradient w.r.t bias
        """

        # for time in reversed(range(error_tensor.shape[0])):
        #     un_sig = self.sigmoid_values[time].backward(error_tensor[time])
        #     self.gradient_b+=un_sig
        #
        #     #self.gradient_why+=np.dot(un_sig.T,self.hidden_values[time].T)
        #     print(np.dot(un_sig,self.hidden_values[time]))
        #     #self.gradient_why+=np.matmul(self.input_tensor[time].T,error_tensor[time])
        #     if(time==error_tensor.shape[0]-1):
        #         self.gradient_hidden=np.matmul(self.weights_hy.T,un_sig)
        #     else:
        #         self.gradient_hidden=np.matmul(self.weights[:,self.input_size:-1].T,self.tanh_values[time].backward(self.gradient_hidden))+ np.matmul(self.weights_hy.T,un_sig)
        #     self.gradient_w[time]+=np.matmul(self.tanh_values[time].backward(self.gradient_hidden),self.concatenated[time])
        #     prev_error[time]=np.matmul(self.tanh_values[time].backward(self.gradient_hidden),self.weights)
        #     print("this iteration: "+str(time))
        #     print(un_sig)
        # print(self.gradient_why.shape)
        # print(self.hidden_values)
        #print(self.hidden_values[0])
        for time in reversed(range(error_tensor.shape[0])):
            un_sig = self.sigmoid_values[time].backward(error_tensor[time])
            self.gradient_b+=un_sig
            #un_sig:output size
            #hidden state? hidden values?
            self.gradient_why+=np.array((np.matmul(np.asmatrix(self.hidden_values[time]).T,np.asmatrix(un_sig))))

            if(time==error_tensor.shape[0]-1):
                self.gradient_hidden=np.zeros(self.hidden_size)

            #un_sig=np.dot(self.weights_hy,error_tensor[time])
            un_sig = np.dot(self.weights_hy,un_sig.T)
            #???
            #un_sig=un_sig +self.gradient_hidden


            un_tan = self.tanh_values[time].backward(un_sig)

            self.gradient_w += np.dot(np.asmatrix(self.concatenated[time]).T,np.asmatrix(un_tan))
            un_tan_concatenatedx = np.matmul(np.asmatrix(un_tan),self.w.T)

            prev_error=un_tan_concatenatedx[0:self.input_size]
            self.gradient_hidden = un_tan_concatenatedx[self.input_size:-1]



        # self.gradient_h=0
        # self.gradient_w=np.zeros((error_tensor.shape[0]))
        # #self.gradient_w=0
        # self.gradient_w= np.zeros((error_tensor.shape[0]))
        # self.gradient_b=np.zeros((prev_error.shape))
        # self.gradient_hy=np.zeros((error_tensor.shape[0],13,self.hidden_state.shape[0]))
        # #self.gradient_w=np.matmul(self.input_tensor.T,error_tensor)
        # print(self.hidden_size)
        # for time in reversed(range(error_tensor.shape[0])):
        #     #self.gradient_b[time] += np.sum(error_tensor[time])
        #     un_sig = self.sig.backward(self.sigmoid_values[time])
        #     if(time==error_tensor.shape[0]-1):
        #         self.gradient_h=np.zeros((self.hidden_size,1))
        #     tmprev_error=un_sig+self.gradient_h
        #     prev_error[time]=np.matmul(un_sig,error_tensor[time])
        #     print(self.hidden_state.shape)
        #     self.gradient_hy[time]=np.dot(prev_error[time],self.hidden_values[time].T)
        #     self.gradient_b[time]=prev_error[time]
        #
        #     #self.gradient_w[time]=np.dot(self.input_tensor[time].T,error_tensor[time])
        #     # print(self.sigmoid_values[time].shape)
        #     # print(self.tanh_values[time].shape)
        #     #un_tanh = self.tan.backward(tmprev_error*self.tanh_values[time])
        #     #print(un_tanh.shape)
        #     # print(tmprev_error.shape)
        #     # print("this iteration is: "+str(time))
        #    # self.gradient_w[time]+=np.sum(np.matmul(self.input_tensor.T,error_tensor))
        #     # print(un_tanh)`


        return prev_error
