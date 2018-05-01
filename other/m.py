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


def norm_corr(window, fil):
    return np.sum( window * fil ) / np.sqrt( np.sum(window**2) * np.sum(fil**2) )


def corr(window, fil):
    return np.sum( window * fil )


def apply_filter(x,fil):
    ilim = int((fil.shape[0]-1)/2)
    jlim = int((fil.shape[1]-1)/2)
    result=np.zeros(x.shape)
    
    padded = np.pad(x,max(ilim,jlim),'reflect')
    plt.imshow(padded, cmap='gray')
    plt.show()

    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            window = get_window(padded,i+jlim,j+jlim,fil.shape)
            result[i,j] = norm_corr(window, fil)
    
    return result


image = cv2.imread('reduc.jpg')
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.imread('res.jpg')
b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
b = b[:,:399]

final = apply_filter(to_float(a), to_float(b))

plt.imshow(a, cmap='gray')
plt.show()
plt.imshow(final, cmap='gray')
plt.show()
