import os
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rn
from skimage import transform

#load the names of image files
image_List = os.listdir('exercise_data')
all_img = np.array([file for file in image_List if file.endswith('.npy')])

#load the labels json file
with open("Labels.json", "r") as file:
    all_labels = json.load(file)

class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

total = range(0,len(all_img))
batch_size = 10
shuffle = True
mirroring = True
rotation = True
size=20 #image size

#array to store labels for output batch
label_batch=[]

#shuffling the whole order of sequences
if shuffle:
    all_img = np.random.permutation(all_img)

if ((len(all_img) % batch_size) == 0):
   #first, compute the batch using file name
   tempBatchNumber = int(len(all_img)/batch_size)
   tempImagesList  = all_img.reshape(tempBatchNumber,batch_size)
   #batch of images
   batch_img = []
   #second, load the appropriate images and labels for batches
   for batch in tempImagesList:
      temp_batch=[]
      temp_label=[]
      for img in batch:
         a = np.load('exercise_data/' + img)
         #resize
         a=transform.resize(a,(size,size))
         temp_batch.append(a)

         fileName = img.split(".")

         temp_label.append(class_dict[all_labels[fileName[0]]])

      batch_img.append(np.array(temp_batch))
      label_batch.append(np.array(temp_label))


else:
   #first, compute atches using file names
   #in this case, create one more additional batch and fill them with files from the beginning
    tempBatchNumber     = math.floor(len(all_img)/batch_size)
    remainder           = (len(all_img) % batch_size)
    itemsToAdd          = batch_size - remainder
    additionalImageList = np.array(all_img[0:itemsToAdd])
    tempImagesList      = np.concatenate((all_img, additionalImageList))
    tempImagesList      = tempImagesList.reshape(tempBatchNumber + 1, batch_size)
    batch_img = []
   #load the appropriate batches and labels
    for batch in tempImagesList:
       temp_batch = []
       temp_label=[]
       for img in batch:
         a = np.load('exercise_data/' + img)
         a = transform.resize(a, (size, size))
         temp_batch.append(a)

         fileName = img.split(".")
         temp_label.append(class_dict[all_labels[fileName[0]]])

       batch_img.append(np.array(temp_batch))
       label_batch.append(np.array(temp_label))


#mirroring
if mirroring:
   temp_batch = []
   for batch in batch_img:
      rotated = []
      for img in batch:
         rotated.append(np.fliplr(img))
      temp_batch.append(rotated)
   batch_img = np.array([])
   batch_img = np.copy(np.array(temp_batch))

#rotation
if rotation:
  # degree = rn.choice([90, 180, 270])
  # transform.rotate(batch_img, degree)
   temp_batch=[]
   for batch in batch_img:
      rotated=[]
      for img in batch:
         degree = rn.choice([90, 180, 270])
         rotated.append(transform.rotate(img,degree))
      temp_batch.append(rotated)
   batch_img=np.array([])
   batch_img =np.copy(np.array(temp_batch))

numberOfItemsEachRow = 3
numberOfRows = math.ceil(batch_size / 3)
figure = plt.figure(figsize=(numberOfItemsEachRow + 2, numberOfRows + 4))

for (index, filePath) in enumerate(batch_img[0]):
    figure.add_subplot(numberOfRows,3,index+1)
    plt.axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.title(label_batch[0][index])
    plt.imshow(batch_img[0][index])

plt.show()
