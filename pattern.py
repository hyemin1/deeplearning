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

#c = Circle(1024, 200, (512, 256))
#c.draw()
#c.show()
