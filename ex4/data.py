import torchvision.transforms
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
from PIL import Image
import skimage
train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self,data,mode):
        super().__init__()
        #img name,cracks, inactive infos
        self.data=data

        #train or val
        self.mode=mode
        #different tranformations for training & validation
        if(self.mode=="val"):
            self._transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=train_mean, std=train_std)])
        else:
            #added random transformation for dta augmentation
            self._transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToPILImage(),torchvision.transforms.RandomRotation(90),torchvision.transforms.RandomGrayscale(),torchvision.transforms.RandomVerticalFlip(),
                 torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=train_mean, std=train_std)])



    def __len__(self):
        #number of files
        return len(self.data)
    def __getitem__(self, index):

        #name of image file
        rt= self.data.iloc[index,0]
        img=imread(rt)
        #RGB coloring
        #apply transformation
        img = skimage.color.gray2rgb(img)
        img = self._transform(img)

        #get the label of the current image
        label = np.array(self.data.iloc[index, 1:],dtype='float')


        #return image and labels
        item=(torch.tensor(img),torch.tensor(label))

        return item

