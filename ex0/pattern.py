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
        result = (self.distance_sqaure<=(self.radius**2))
        self.output=np.copy(result)
        return result

    def show(self):
        #plot the circle with the output array
        plt.imshow(self.output, cmap='gray')
        plt.show()

class Checker:
    def __init__(self, resolution, tile_size):
        #Width and Height of the checkerboard
        self.resolution = resolution
        #Size of every tile
        self.tile_size = tile_size

    def draw(self):
        if (self.resolution % (2 * self.tile_size)):
            print('Error: self.resolution % (2 * self.tile_size) must be an integer')
            return False

        #Creating an array of zeros in order to represent the white tiles
        arrayOfZeroes = np.zeros(self.tile_size)

        #Creating an array of ones in order to represent the black tiles
        arrayOfOnes = np.ones(self.tile_size)

        #Joining an array of zeroes and ones via np.concatenate
        concatenated_array = np.concatenate((arrayOfZeroes,arrayOfOnes))

        # We are adding padding to the array since we need to create all the elements for the blacks and whites
        # By (self.resolution**2) we are calculating the area of the board
        # Since we have generated the area of the board with two element at once. In order to balance that we are deviding it by two (self.resolution**2)/2
        # Now we are deducting the initial element we have taken into our account (self.resolution**2)/2-self.tile_size)
        padded_array = np.pad(concatenated_array,int((self.resolution**2)/2-self.tile_size),'wrap')

        # After padding the array we are reshaping the array into Resolution * Resolution size
        square_reshaped_array = padded_array.reshape((self.resolution,self.resolution))

        # Now there is a problem. The square shaped array has identical elements in each row. But When one row finishes
        # with white tile, the next element on the second row should be black or vise-versa.
        # In order to do that we are using the technique of adding the array with transposed version of the same array.
        # By (T == 1) we are converting the 2D array into true and false and adding to the initial array.
        reshuffled_array = (square_reshaped_array+square_reshaped_array.T == 1)

        # Finally converting the types of variables into integer format in order to plottable via Matplotlib and assigning it to output variable
        result = reshuffled_array.astype(int)
        self.output= np.copy(result)
        return result

    def show(self):
        #Showing the output via matplotlib
        plt.imshow(self.output, cmap = "gray")
        plt.show()

class Spectrum:
    def __init__(self, resolution):
        #Resoulution of Color Spectrum
        self.resolution = resolution

    def draw(self):
        #Creating an array where the number of elements is equal to resolution. The array range from 0 to number of resolution.
        rgbRangeArray = np.arange(0, self.resolution)

        #Crearing an array with ones. The number of elements in the array is equal to the resolution.
        onesArray = np.ones(self.resolution, dtype=np.uint8)

        #Now we arae multiplying this two array with np.outer function
        multipliedArray = np.outer(rgbRangeArray, onesArray)

        #We are creating an three dimentional array with the values of resolution
        zeroRgbArray = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        #Changing the value of the first row with multiplied array
        zeroRgbArray[:, :, 0] = multipliedArray

        #Changing the value of the second row with the transpose of the multiplied array
        zeroRgbArray[:, :, 1] = multipliedArray.T

        #Assigning to the output
        self.output = zeroRgbArray
        return self.output

    def show(self):
        #Showing the output via Matplotlib
        plt.imshow(self.output, origin='lower', interpolation='nearest')
        plt.show()
