import os.path
import json
import scipy.misc
import numpy as np
import skimage
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        #pictures=np.arrays('exercise_data/'+range(0,100)+'.npy')
        #self.all_images = np.arrays(np.load(file) for file in pictures)
        self.all_images=np.array()
        self.all_labels=np.array()
        #not sure about this. maybe have to change the fixed range
        for i in range(0,100):
            file_num=str(i)
            self.all_images.append(np.load('exercise_data/'+file_num+'.npy'))
            self.all_labels.append(file_num)
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        #do we have to get several batches of the whole array?
        if(len(self.all_images)%2!=0):
            print()
        else:
            #arrays for all batches and labels
            self.total_batch=np.array((len(self.all_images)/self.batch_size,self.batch_size))
            self.total_labels=np.array((len(self.all_images)/self.batch_size,self.batch_size))

            #temporal batches and labels to append at total batch and labels
            temp_batch=np.array()
            temp_label=np.array()
            #for all images
            for i in range(0,len(self.all_images)):
                #append image and label to temporal array
                temp_batch.append(self.all_images[i])
                temp_label.append(self.all_labels[i])
                #if the length of temporal array is same as given batch size
                if(len(temp_batch)==self.batch_size):
                    #append to total batch and labels array
                    self.total_batch.append(temp_batch)
                    self.total_labels.append(temp_label)
                    #clear temporal arrays
                    temp_batch.empty()
                    temp_label.empty()
            #resize all images according to the given image size
            self.total_batch=skimage.transform.resize(self.total_batch,self.image_size,self.image_size)
        #return first batch of images and labels
        images = self.total_batch[0]
        labels = self.total_labels[0]
       # images = np.array(self.all_images[i] for i in range(self.batch_size))
        #images = skimage.transform.resize(images,self.image_size,self.image_size)
        #labels = np.arrays(self.class_dict[i] for i in range(self.batch_size))
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
        return
    def show(self):
        print()
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

