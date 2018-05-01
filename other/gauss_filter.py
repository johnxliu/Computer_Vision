from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np
image = cv2.imread('a.jpeg')
a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def add_noise(x):
    noise=np.random.normal(0, 50, x.shape)
    return cv2.add(x, to_uint8(noise))

def to_uint8(x): 
    x=np.clip(x, 0, 255, out=x)
    return x.astype('uint8')

def to_float(x): return x.astype('float')


def gaussian_filter(size, sigma):
    lim=int((size-1)/2)
    def gauss(x,sigma): return np.exp( -(x)**2/ (2*sigma**2) )
    #def gauss(x,y,sigma): return np.exp( -(x**2+y**2)/ (2*sigma**2) )
    #def gauss(x,sigma): return 1/np.sqrt(2*np.pi*sigma**2)*np.exp( -(x)**2/ (2*sigma**2) )
    x=np.arange(-lim,lim+1)
    fil=np.outer(gauss(x,sigma),gauss(x,sigma))
    fil=fil/np.sum(fil)
    #x,y=np.meshgrid(x,x)
    #fil=gauss(x,y,sigma)
    print fil
    return fil


def get_window(x,i,j,size):
    side=int((size-1)/2)
    return x[i-side:i+1+side,j-side:j+1+side]


def apply_filter(x,fil):
    lim=int((fil.shape[0]-1)/2)

    for i in range(lim,x.shape[1]-lim):
        for j in range(lim,x.shape[0]-lim):
           x[i,j]=np.sum(get_window(x,i,j,fil.shape[0])*fil)
           print x[i,j]

    return x


def smooth(x,size):
    fil=np.ones((size,size))/size**2
    x=apply_filter(x,fil)
    return x

x=np.zeros((11,11))
x[5,5]=1
print x

print apply_filter(x,np.array([ [2,0,0],[0,1,0],[3,0,0]]))
#
#
#
#final=to_uint8(apply_filter(to_float(a),np.array([ [0,0,0],[0,1,0],[0,0,0]])))
#
#
#cv2.imshow('gray_image', a)
#cv2.waitKey(0)                 # Waits forever for user to press any key
#cv2.imshow('gray_image', final)
#cv2.waitKey(0)                 # Waits forever for user to press any key
