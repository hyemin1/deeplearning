import os
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import random as rn

image_List = os.listdir('/Users/rupak/Documents/Deep Learning Exersises/exercise0_material/src_to_implement/data/exercise_data')
allImagesList = np.array([file for file in image_List if file.endswith('.npy')])

with open("/Users/rupak/Documents/Deep Learning Exersises/exercise0_material/src_to_implement/data/Labels.json", "r") as file:
    all_labels = json.load(file)

class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

total = range(0,len(allImagesList))
batch_size = 16
shuffle = True

if ((len(allImagesList) % batch_size) == 0):
    tempBatchNumber = int(len(allImagesList)/batch_size)
    tempImagesList  = allImagesList.reshape(tempBatchNumber,batch_size)
else:
    tempBatchNumber     = math.floor(len(allImagesList)/batch_size)
    remainder           = (len(allImagesList) % batch_size)
    itemsToAdd          = batch_size - remainder
    additionalImageList = np.array(allImagesList[0:itemsToAdd])
    tempImagesList      = np.concatenate((allImagesList, additionalImageList))
    tempImagesList      = tempImagesList.reshape(tempBatchNumber + 1, batch_size)

if shuffle:
    tempImagesList = np.random.permutation(tempImagesList)

path = "/Users/rupak/Documents/Deep Learning Exersises/exercise0_material/src_to_implement/data/exercise_data/"

imagePathArray = np.array([f"{path}{file}" for file in tempImagesList[0]])
imagesLabelArray = []

for fileName in tempImagesList[0]:
    fileNumber = fileName.split(".")
    imagesLabelArray.append(class_dict[all_labels[fileNumber[0]]])

figure = plt.figure(figsize=(8, 8))

for (index, filePath) in enumerate(imagePathArray):
    figure.add_subplot(4,4,index+1)
    plt.imshow(np.load(imagePathArray[index]))

plt.show()
