from __future__ import division
import cv2
import matplotlib.pyplot as plt
import numpy as np




#Auxiliary functions

def to_uint8(x): 
    x = np.clip(x, 0, 255, out=x)
    return x.astype('uint8')


def to_float(x): return x.astype('float')


def get_window(x,i,j,shape):
    ilim = int((shape[0]-1)/2)
    jlim = int((shape[1]-1)/2)
    return x[i-ilim:i+1+ilim,j-jlim:j+1+jlim]


#Filters


def norm_corr(window, fil):
    return np.sum( window * fil ) / np.sqrt( np.sum(window**2) * np.sum(fil**2) )


def corr(window, fil):
    return np.sum( window * fil )


def apply_filter(x, fil):
    ilim = int((fil.shape[0]-1)/2)
    jlim = int((fil.shape[1]-1)/2)
    result=np.zeros(x.shape)

    # Could be modified
    padded = np.pad(x,max(ilim,jlim),'reflect')

    for i in range(0,x.shape[0]):
        for j in range(0,x.shape[1]):
            window = get_window(padded,i+jlim,j+jlim,fil.shape)
            result[i,j] = corr(window, fil)
    
    return result


def smooth(x,size):
    fil=np.ones((size,size))/size**2
    x=apply_filter(x,fil)
    return x


def add_noise(x):
    noise = np.random.normal(0, 50, x.shape)
    return cv2.add(x, to_uint8(noise))


def gaussian_2D(size, sigma):
    lim = int((size-1)/2)

    def gauss(x,sigma): return np.exp( -(x)**2/ (2*sigma**2) )

    #def gauss(x,y,sigma): return np.exp( -(x**2+y**2)/ (2*sigma**2) )
    #def gauss(x,sigma): return 1/np.sqrt(2*np.pi*sigma**2)*np.exp( -(x)**2/ (2*sigma**2) )

    x = np.arange(-lim,lim+1)
    fil = np.outer( gauss(x,sigma), gauss(x,sigma) )
    fil = fil / np.sum(fil)
    return fil


def gradient_2D(x):
    sx = np.array([[-1, 0 , 1], [-2, 0, 2], [-1, 0, 1]])/8
    sy = np.array([[1, 2 , 1], [0, 0, 0], [-1, -2, -1]])/8

    gx = apply_filter(to_float(x),sx)
    gy = apply_filter(to_float(x),sy)

    normsqr = gx**2 + gy**2
    ang = np.arctan2(gx,gy)
    return normsqr, ang


#def apply_filter(x,fil):
#    ilim = int((fil.shape[0]-1)/2)
#    jlim = int((fil.shape[1]-1)/2)
#    result=np.zeros(x.shape)
#
#    padded = np.pad(x,max(ilim,jlim),'reflect')
#
#    for i in range(ilim,x.shape[0]-ilim):
#        for j in range(jlim,x.shape[1]-jlim):
#            window = get_window(x,i,j,fil.shape)
#            result[i,j] = norm_corr(window, fil)
#
#    return result


