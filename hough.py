from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import *

image = cv2.imread('bn.jpg')
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#complete
def find_peaks( x ):
    return np.nonzero( x )

def hough( x, angle_bins ):

    dmax = int( np.ceil( np.sqrt( np.sum( x.shape[0]**2 + x.shape[1]**2 ) ) ) )
    n_angle = 90 / angle_bins  #in degrees
    #find limit
    angles = np.arange(0, np.pi/2, np.deg2rad( angle_bins ))
    
    #initialize 
    H = np.zeros( (dmax, len(angles)) )

    peaks = find_peaks( x )
    cosang = np.cos(angles)
    sinang = np.sin(angles)

    #try to vectorize
    for peak in peaks:
        for i in range(len(angles)):
            rho = peak[1]*cosang[i] + peak[0]*sinang[i]
            print rho
            H[int(np.round(rho)), i] += 1
    return H



canny = cv2.Canny(a,70,140)
canny = to_float( canny )
a = hough(canny, 10)
print np.amax(a)

plt.imshow(a, cmap='gray')
plt.show()
#sobelx = cv2.Sobel(a,cv2.CV_64F,1,0,ksize=1)/8
