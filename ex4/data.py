import torchvision.transforms
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self,data,mode):
        super().__init__()
        self.data=data
        self.mode=mode
        if(self.mode=="train"):
            self._transform=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=train_mean,std=train_std)])
        else:
            self._transform=None
    def __len__(self):
        return len(self.data.loc[:,0])
    def __getitem__(self, index):
        #maybe have to add reading image using skimage
        print(self.data.loc[0])
        img,label1,label2=self.data.loc[index,:]
        img =imread(img)
        img=gray2rgb(img)
        if(self.mode=="train"):
            img=self._transform(img)
        return torch.tensor(img,label1,label2)
        pass
