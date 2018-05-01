from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np


def to_uint8(x): 
    x = np.clip(x, 0, 255, out=x)
    return x.astype('uint8')


def to_float(x): return x.astype('float')


def get_window(x,i,j,shape):
    ilim = int((shape[0]-1)/2)
    jlim = int((shape[1]-1)/2)
    return x[i-ilim:i+1+ilim,j-jlim:j+1+jlim]


def apply_filter(x,fil):
    ilim = int((fil.shape[0]-1)/2)
    jlim = int((fil.shape[1]-1)/2)
    result=np.zeros(x.shape)

    for i in range(ilim,x.shape[0]-ilim):
        for j in range(jlim,x.shape[1]-jlim):
            window = get_window(x,i,j,fil.shape)
            result[i,j] = np.sum( window * fil ) / np.sqrt( np.sum(window**2) * np.sum(fil**2) )

    return result


image = cv2.imread('search.jpg')
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
a=a[:,:619]

image = cv2.imread('wally.jpg')
b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print a.shape

final = apply_filter(to_float(a),to_float(b))
max_ind = np.unravel_index(final.argmax(), final.shape)
print final
#final = to_uint8( final  )
plt.plot(max_ind[1],max_ind[0],'ro')
plt.imshow(a, cmap='gray')
plt.show()
plt.imshow(final, cmap='gray')#,vmin=0,vmax=1)
plt.show()

#cv2.imshow('gray_image', final)
#cv2.waitKey(0)                 # Waits forever for user to press any key
