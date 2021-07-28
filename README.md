# deeplearning
problem solving for deep learning course, summer semester 2021, FAU

ex0- Numpy warm up
1. checkerboard
- build a checkerboard with adaotable size and resolution
2. Circle
- draw a white circle with black background
3. RGB Spectrum
- draw a RGB Spectrum with different colors at each corner 


ex1-Basic Framework
- Layer: Fully Connected Layer, Base
- Activation Function: ReLU
- Loss: Cross Entropy Loss with Softmax Function
- Optimizer: Stochastic Gradient Descent(SCD)


ex2-CNN
- Layer:Convolutional Neural Network(CNN), Flatten, FC, Pooling, Base
- Initializer: Constant, UniformRandom, Xavier, He
- Optimizer: SGD With Momentom, Adam
- Activation Function: ReLU
- Loss: Cross Entropy Loss with Softmax Function

ex3-Regularization, RNN
- Layer:FC,Recurrent Neural Network(RNN),Pooling,Flatten,Base
- Regularization(layer):Drop Out,BatchNormalization
- Regularization(Constraints): L1 Regularizer, L2 Regularizer
- Activation Functions: ReLU,Sogmoid,TanH
- Loss:Cross Entropy Loss with Softmax Function

ex4-PyTorch Challenge
- train the ResNet using pytorch to check the cracks and inactive regions in solar pannels.Use GPU to train the model.
- Architecture: ResNet
- batch size: 36
- learning rate: 0.00007
- total epoch:200
- test size:0.2
- early stopping patience: 20
- Loss: BCE Loss 
- Optimizer: Adam

RESULT
- f1 score of cracks: 0.63636
- f1 score of inactive regions: 0.76923
- f1 mean:	0.70280



