import torch
import torch as t
from sklearn.metrics import f1_score
import numpy
from tqdm.autonotebook import tqdm
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self.avg_pred = []
        self.val_y = []
        self._if_cuda=cuda
        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._model.zero_grad()
        # -propagate through the network
        out = self._model(x)
        predict = out.to(torch.float)
        y = y.to(torch.float)

        loss = self._crit(predict, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss


    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        out = self._model(x)
        predict = out.to(torch.float)
        y = y.to(torch.float)

        loss = self._crit(predict, y)
        # return the loss and the predictions
        return loss, predict


    def train_epoch(self):
        # set training mode
        self._model.train()
        # iterate through the training set
        epoch_loss = 0
        total = 0


        for item in (self._train_dl):
            # transfer the batch to "cuda()" -> the gpu if a gpu is given

            img = item[0]
            label = item[1]
            # perform a training step

            img=img.cuda()
            label=label.cuda()

            epoch_loss += self.train_step(img, label)

            total += 2  # add 2 (number of classes of prediction)

        # calculate the average loss for the epoch and return it

        epoch_loss = epoch_loss / total
        return epoch_loss


    def val_test(self):

        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.

        total = 0
        total_loss = 0
        prediction_array = []
        label_array = []
        with torch.no_grad():
            # iterate through the validation set
            for item in (self._val_test_dl):
                # transfer the batch to the gpu if give
                img = item[0]
                label = item[1]

                # perform a validation step
                img = img.cuda()
                label = label.cuda()

                val_loss, out = self.val_test_step(img, label)

                # save the predictions and the labels for each batch
                total += 2
                total_loss += val_loss

                out = out.cpu().data.numpy()
                label = label.cpu().data.numpy()
                out = out.flatten()
                label = label.flatten()

                out[out > 0.5] = 1
                out[out <= 0.5] = 0

                prediction_array.append(out)
                label_array.append(label)

        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        total_loss = total_loss / total

        prediction_array = numpy.array(prediction_array).flatten()
        label_array = numpy.array(label_array).flatten()

        print(f1_score(y_true=label_array, y_pred=prediction_array, average='micro'))

        return total_loss
        # TODO

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        self.train_loss = []
        self.val_loss = []
        counter = 0
        early_counter = 0

        while True:
            # stop by epoch number
            if (counter == epochs):
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            self.train_loss.append(self.train_epoch())
            # append the losses to the respective lists
            temp = self.val_test()
            self.val_loss.append(temp)
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # if (counter%5==0):
                # self.save_checkpoint(epochs)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if (temp < 0.13):
                early_counter += 1
            if (early_counter == self._early_stopping_patience):
                break
            counter += 1

        # return the losses for both training and validation
        return self.train_loss, self.val_loss














# import torch
# import torch as t
# from sklearn.metrics import f1_score
# from tqdm.autonotebook import tqdm
#
#
# class Trainer:
#
#     def __init__(self,
#                  model,                        # Model to be trained.
#                  crit,                         # Loss function
#                  optim=None,                   # Optimizer
#                  train_dl=None,                # Training data set
#                  val_test_dl=None,             # Validation (or test) data set
#                  cuda=True,                    # Whether to use the GPU
#                  early_stopping_patience=-1):  # The patience for early stopping
#         self._model = model
#         self._crit = crit
#         self._optim = optim
#         self._train_dl = train_dl
#         self._val_test_dl = val_test_dl
#         self._cuda = cuda
#         self.avg_pred=[]
#         self.val_y=[]
#
#         self._early_stopping_patience = early_stopping_patience
#
#         if cuda:
#             self._model = model.cuda()
#             self._crit = crit.cuda()
#
#     def save_checkpoint(self, epoch):
#         t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
#
#     def restore_checkpoint(self, epoch_n):
#         ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
#         self._model.load_state_dict(ckp['state_dict'])
#
#     def save_onnx(self, fn):
#         m = self._model.cpu()
#         m.eval()
#         x = t.randn(1, 3, 300, 300, requires_grad=True)
#         y = self._model(x)
#         t.onnx.export(m,                 # model being run
#               x,                         # model input (or a tuple for multiple inputs)
#               fn,                        # where to save the model (can be a file or file-like object)
#               export_params=True,        # store the trained parameter weights inside the model file
#               opset_version=10,          # the ONNX version to export the model to
#               do_constant_folding=True,  # whether to execute constant folding for optimization
#               input_names = ['input'],   # the model's input names
#               output_names = ['output'], # the model's output names
#               dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                             'output' : {0 : 'batch_size'}})
#
#     def train_step(self, x, y):
#         # perform following steps:
#         # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
#         # self._crit.zero_grad()
#         self._model.zero_grad()
#         # -propagate through the network
#         out = self._model(x)
#         predict=out.to(torch.float)
#         y=y.to(torch.float)
#
#         loss=self._crit(predict,y)
#         # -compute gradient by backward propagation
#         loss.backward()
#         # -update weights
#         self._optim.step()
#         # -return the loss
#         return loss
#         #TODO
#
#
#
#     def val_test_step(self, x, y):
#         """
#         check
#         """
#         # predict
#         # propagate through the network and calculate the loss and predictions
#         out = self._model(x)
#         predict = out.to(torch.float)
#         y = y.to(torch.float)
#
#
#         loss = self._crit(predict, y)
#         # return the loss and the predictions
#         return loss,predict
#         #TODO
#
#     def train_epoch(self):
#         # set training mode
#         self._model.train()
#         # iterate through the training set
#         epoch_loss=0
#         total=0
#         i=0
#         cnt=0
#
#         for item in (self._train_dl):
#             # transfer the batch to "cuda()" -> the gpu if a gpu is given
#             """
#             add cuda to use GPU
#             """
#             img=item[0]
#             label=item[1]
#             # perform a training step
#             img=img.cuda()
#             label=label.cuda()
#
#
#             epoch_loss+=self.train_step(img,label)
#
#             total+=2 #add 2 (number of classes of prediction)
#
#         # calculate the average loss for the epoch and return it
#
#         epoch_loss=epoch_loss/total
#         return epoch_loss
#         #TODO
#
#     def val_test(self):
#
#         # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
#         self._model.eval()
#         # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
#         right=0
#         total=0
#         total_loss=0
#         cnt=0
#         tmp_pred=[]
#         tmp_label=[]
#         with torch.no_grad():
#             # iterate through the validation set
#             for item in (self._val_test_dl):
#                 # transfer the batch to the gpu if given
#                 """
#                 add cuda to use GPU
#                 """
#                 img=item[0]
#                 label=item[1]
#
#                 # perform a validation step
#                 img=img.cuda()
#                 label=label.cuda()
#                 cnt+=1
#                 val_loss,out=self.val_test_step(img,label)
#
#                 # save the predictions and the labels for each batch
#                 total+=2
#                 total_loss+=val_loss
#                 tmp_pred.append(out)
#                 tmp_label.append(label)
#                 """
#                 should add something
#                 """
#         # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
#         # tmp_pred=tmp_pred/total
#         # tmp_label=tmp_label/total
#         tmp_pred=[x/total for x in tmp_pred]
#         tmp_label=[x/total for x in tmp_label]
#         self.avg_pred.append(tmp_pred)
#         self.val_y.append(tmp_label)
#         total_loss=total_loss/total
#         # return the lossand print the calculated metrics
#         # tmp_label=tmp_label.to(torch.int)
#         # tmp_pred=tmp_pred.to(torch.int)
#
#         # print("f1 score")
#         # print((self.avg_pred))
#         # print((self.val_y))
#         # print(f1_score(y_true=tmp_label, y_pred=tmp_pred, average='micro'))
#
#         return total_loss
#         #TODO
#
#
#     def fit(self, epochs=-1):
#         assert self._early_stopping_patience > 0 or epochs > 0
#         # create a list for the train and validation losses, and create a counter for the epoch
#         self.train_loss=[]
#         self.val_loss=[]
#         counter=0
#         early_counter=0
#         #TODO
#
#         while True:
#             # stop by epoch number
#             if(counter==epochs):
#                 break
#             # train for a epoch and then calculate the loss and metrics on the validation set
#             self.train_loss.append(self.train_epoch())
#             # append the losses to the respective lists
#             temp=self.val_test()
#             self.val_loss.append(temp)
#             # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
#             # self.save_checkpoint(epochs)
#             # check whether early stopping should be performed using the early stopping criterion and stop if so
#             if(temp<0.13):
#                 early_counter+=1
#             if (early_counter== self._early_stopping_patience):
#                 break
#             counter+=1
#
#
#         # return the losses for both training and validation
#         return self.train_loss,self.val_loss
#         #TODO
                    
        
        
        
