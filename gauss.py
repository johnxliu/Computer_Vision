from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import *

image = cv2.imread('lena.jpg')
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#
g = gaussian_2D( 10, 1)
a = cv2.blur(a, (3,3))

#derx  = cv2.Sobel(a,cv2.CV_64F,1,0,ksize=3)/8
#der2x  = cv2.Sobel(derx,cv2.CV_64F,1,0,ksize=3)/8
#dery  = cv2.Sobel(a,cv2.CV_64F,0,1,ksize=3)/8
#der2y  = cv2.Sobel(dery,cv2.CV_64F,0,1,ksize=3)/8
#lap = derx**2 + dery**2
#print np.max(lap)
#lap[lap<100] = 0

final = cv2.Canny(a,70,140)

plt.imshow(final, cmap='gray')
plt.show()

#from mpl_toolkits.mplot3d.axes3d import Axes3D
#
#fig = plt.figure()#figsize=(14,6))
#
##ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax = Axes3D(fig)
#
#print final.shape
#y , x = np.meshgrid( range(final.shape[1]), range(final.shape[0]) )
#p = ax.plot_surface(x, y, final,  rstride=1, cstride=1, linewidth=0, cmap='gray')
#
## surface_plot with color grading and color bar
##ax = fig.add_subplot(1, 2, 2, projection='3d')
##p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.greyscale, linewidth=0, antialiased=False)
##cb = fig.colorbar(p, shrink=0.5)
#plt.show()
