import numpy as np
import matplotlib.pyplot as plt

class Circle:
    def __init__(self,resolution,radius, position):
        #resolution of the grid
        self.resolution=resolution
        #radious of the circle
        self.radius=radius
        #center of the circle
        self.x,self.y = position

    def draw(self):
        #get positions of all pixels
        X_coordinate,Y_coordinate = np.meshgrid(range(self.resolution),range(self.resolution))
        #compute sqaured distance from the given center
        self.distance_sqaure = (X_coordinate-self.x)**2 +(Y_coordinate-self.y)**2
        #set boolean value: if the point is inside of the circle or not
        #circle equation: x**2 + y**2 = r**2
        self.output = (self.distance_sqaure<=(self.radius**2))
        return self.output

    def show(self):
        #plot the circle with the output array
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size

    def draw(self):
        if (self.resolution % (2 * self.tile_size)):
            print('Error: self.resolution % (2 * self.tile_size) must be an integer')
            return False
        concatenated_array = np.concatenate((np.zeros(self.tile_size),np.ones(self.tile_size)))
        pattern = np.pad(concatenated_array,int((self.resolution**2)/2-self.tile_size),'wrap').reshape((self.resolution,self.resolution))
        self.output = (pattern+pattern.T==1).astype(int)
        return self.output

    def show(self):
        plt.imshow(self.output, cmap = "gray")
        plt.show()
