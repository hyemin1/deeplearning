import os
import json
import matplotlib.pyplot as plt
import numpy as np
import math

image_List = os.listdir('/Users/rupak/Documents/Deep Learning Exersises/exercise0_material/src_to_implement/data/exercise_data')
allImagesList = np.array([file for file in image_List if file.endswith('.npy')])

with open("/Users/rupak/Documents/Deep Learning Exersises/exercise0_material/src_to_implement/data/Labels.json", "r") as file:
    all_labels = json.load(file)

class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

total = range(0,len(allImagesList))
batch_size = 11

if ((len(allImagesList) % batch_size) == 0):
    tempBatchNumber = int(len(allImagesList)/batch_size)
    tempImagesList  = allImagesList.reshape(batch_size, tempBatchNumber)
else:
    tempBatchNumber = math.floor(len(allImagesList)/batch_size)
    remainder       = (len(allImagesList) % batch_size)
    itemsToAdd      = batch_size - remainder
    elementCount    = len(allImagesList) + itemsToAdd
    tempImageList   = allImagesList[0:itemsToAdd]
    print (np.concatenate(np.array(allImagesList),np.array(tempImageList)))
