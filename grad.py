from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
from functions import *

image = cv2.imread('stop.jpg')
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sx = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])/8
sy = np.array([[1, 2 , 1], [0, 0, 0], [-1, -2, -1]])/8

gx = apply_filter(to_float(a),sx)/4
sobelx = cv2.Sobel(a,cv2.CV_64F,1,0,ksize=1)/8
gy = apply_filter(to_float(a),sy)

normsqr = gx**2 + gy**2
ang = np.arctan2(gx,gy)
final = ang

plt.imshow(a, cmap='gray')
plt.show()
plt.imshow(normsqr, cmap='gray')#,vmin=0,vmax=1)
plt.show()
plt.imshow(ang, cmap='gray')#,vmin=0,vmax=1)
plt.show()

#cv2.imshow('gray_image', final)
#cv2.waitKey(0)                 # Waits forever for user to press any key
