import os.path
import json
import scipy.misc
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import random
import math


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
        self.batch_num = 0  # to return n-th batch

        # load the names of image files
        image_List = os.listdir(self.file_path)
        self.all_img = np.array([file for file in image_List if file.endswith('.npy')])

        # load the labels json file
        with open(self.label_path, "r") as file:
            self.all_labels = json.load(file)

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

        # array to store labels for output batch
        self.batch_label = []
        self.batch_img = []


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        # temporal batches and labels to append at total batch and labels

        # shuffling the whole order of sequences
        if self.shuffle:
            self.all_img = np.random.permutation(self.all_img)

        if ((len(self.all_img) % self.batch_size) == 0):
            # first, compute the batch using file name
            tempBatchNumber = int(len(self.all_img) / self.batch_size)
            tempImagesList = self.all_img.reshape(tempBatchNumber, self.batch_size)
        else:
            # first, compute batches using file names
            # in this case, create one more additional batch and fill them with files from the beginning
            tempBatchNumber = math.floor(len(self.all_img) / self.batch_size)
            remainder = (len(self.all_img) % self.batch_size)
            itemsToAdd = self.batch_size - remainder
            additionalImageList = np.array(self.all_img[0:itemsToAdd])
            tempImagesList = np.concatenate((self.all_img, additionalImageList))
            tempImagesList = tempImagesList.reshape(tempBatchNumber + 1, self.batch_size)

        # load the appropriate batches and labels
        for batch in tempImagesList:
            temp_batch = []
            temp_label = []
            for img in batch:
                a = np.load('exercise_data/' + img)
                a = transform.resize(a, (self.image_size[0], self.image_size[1]), mode='constant')
                temp_batch.append(a)

                fileName = img.split(".")
                temp_label.append(self.all_labels[fileName[0]])

            self.batch_img.append(np.array(temp_batch))
            self.batch_label.append(np.array(temp_label))

        # mirroring
        if self.mirroring:
            temp_batch = []
            for batch in self.batch_img:
                rotated = []
                for img in batch:
                    rotated.append(np.fliplr(img))
                temp_batch.append(rotated)
            self.batch_img = np.array([])
            self.batch_img = np.copy(np.array(temp_batch))

        # rotation
        if self.rotation:
            temp_batch = []
            for batch in self.batch_img:
                rotated = []
                for img in batch:
                    degree = random.choice([90, 180, 270])
                    rotated.append(transform.rotate(img, degree))
                temp_batch.append(rotated)
            self.batch_img = np.array([])
            self.batch_img = np.copy(np.array(temp_batch))

        self.images = np.copy(self.batch_img[self.batch_num])
        self.labels = np.copy(self.batch_label[self.batch_num])
        images = np.copy(self.images)
        labels = np.copy(self.labels)
        self.batch_num += 1
        return images, labels

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image

        action = np.random(0, 3)
        # rotation
        if action == 0:
            degree = random.choice([90, 180, 270])
            img = transform.rotate(img, degree)
        # mirror
        elif action == 1:
            img = np.fliplr(img)
        # mirroring and rotation
        else:
            degree = random.choice([90, 180, 270])
            img = transform.rotate(img, degree)
            img = np.fliplr(img)

        return img

    def class_name(self, x):
        # This function returns the class name for a specific input

        return self.class_dict[x]


    def show(self):
        print()
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.

        numberOfItemsEachRow = 3
        numberOfRows = math.ceil(self.batch_size / 3)
        figure = plt.figure(figsize=(numberOfItemsEachRow + 2, numberOfRows + 4))

        for index in range(len(self.batch_img[self.batch_num])):
            figure.add_subplot(numberOfRows, 3, index + 1)
            plt.axis('off')
            plt.subplots_adjust(hspace=0.5)
            plt.title(self.class_name(self.batch_label[self.batch_num][0]))
            plt.imshow(self.batch_img[self.batch_num][index])

        plt.show()
