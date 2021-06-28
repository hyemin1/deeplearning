import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

"""
hyper parameters
"""
batch_size=50
learning_rate=0.1
total_epoch=200
# test size for splitting

# load the data from the csv file and perform a train-test-split
data = pd.read_csv("data.csv",sep=';')

# this can be accomplished using the already imported pandas and sklearn.model_selection modules
train,test = train_test_split(data,test_size=0.3)
# TODO

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
val_loader = t.utils.data.DataLoader(ChallengeDataset(test, 'val'), batch_size=batch_size)
train_loader=t.utils.data.DataLoader(ChallengeDataset(train, 'val'), batch_size=batch_size)
# TODO

# create an instance of our ResNet model
resnet= model.ResNet()
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
"""
choose loss function
"""
loss=t.nn.CrossEntropyLoss()
# set up the optimizer (see t.optim)
"""
choose optimizer
"""
opt=t.optim.Adam(resnet.parameters(),lr=learning_rate)

# create an object of type Trainer and set its early stopping criterion
trainer=Trainer(early_stopping_patience=100,model=resnet,crit=loss,optim=opt,train_dl=train_loader,val_test_dl=val_loader)
# TODO
res=[[0],[0]]
# go, go, go... call fit on trainer
res[1],res[0]=trainer.fit(total_epoch)
#TODO

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')