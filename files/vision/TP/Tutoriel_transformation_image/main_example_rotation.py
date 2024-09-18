#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:21:01 2022

@author: guillaume
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

def interp2_bilinear(im, x, y):
    
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    maskx = np.logical_or(x0<0, x1>im.shape[1]-1)
    masky = np.logical_or(y0<0, y1>im.shape[0]-1)
    mask_notvalid = np.logical_or(maskx,masky)
    
    x0[mask_notvalid] = 0
    x1[mask_notvalid] = 0
    y0[mask_notvalid] = 0
    y1[mask_notvalid] = 0
    
    if(len(im.shape)==3):
        Ia = im[ y0, x0, : ]
        Ib = im[ y1, x0, : ]
        Ic = im[ y0, x1, : ]
        Id = im[ y1, x1, : ]
        
        wa = np.expand_dims((x1-x) * (y1-y),axis=Ia.ndim-1)
        wb = np.expand_dims((x1-x) * (y-y0),axis=Ia.ndim-1)
        wc = np.expand_dims((x-x0) * (y1-y),axis=Ia.ndim-1)
        wd = np.expand_dims((x-x0) * (y-y0),axis=Ia.ndim-1)
    
    else:
        Ia = im[ y0, x0]
        Ib = im[ y1, x0]
        Ic = im[ y0, x1]
        Id = im[ y1, x1]
        
        wa = ((x1-x) * (y1-y))
        wb = ((x1-x) * (y-y0))
        wc = ((x-x0) * (y1-y))
        wd = ((x-x0) * (y-y0))


    return wa*Ia + wb*Ib + wc*Ic + wd*Id, mask_notvalid

def getInterpolationGrids(theta, center_rot, h_new, w_new):
    
    theta_rad = theta*math.pi/180.
    rot_mat = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],[np.sin(theta_rad), np.cos(theta_rad)]])
    
    x = np.arange(0,w_new)
    y = np.arange(0,h_new)
    X, Y = np.meshgrid(x,y) #h_new x w_new
    
    p = np.zeros((h_new,w_new,2)) #h_new x w_new x 2
    p[:,:,0] = X
    p[:,:,1] = Y
    
    
    p_rot = ((p - np.array([w_new/2., h_new/2.])) @ rot_mat) + center_rot #h_new x w_new x 2

    XI = p_rot[:,:,0] #h_new x w_new
    YI = p_rot[:,:,1] #h_new x w_new
 

    return XI, YI

#%% load image to transform
I = np.array(Image.open('Tour_eiffel.jpg')).astype(float)/255.
h,w,_ = I.shape
fig2, axs2 = plt.subplots(ncols=2)
axs2[0].imshow(I)
axs2[0].set_title('original image')
plt.pause(0.1)

#%% Define rotation angle
theta = 60 #the resulting image should be rotated 90° right
center_rot = np.array([w/2., h/2.])

#%% Define size of resulting image
h_new = 1280
w_new = 960

#%% get interpolation grids

XI,YI = getInterpolationGrids(theta, center_rot, h_new, w_new)

fig1, axs1 = plt.subplots(ncols=2)
axs1[0].imshow(XI)
axs1[0].set_title('XI')
axs1[1].imshow(YI)
axs1[1].set_title('YI')
plt.pause(0.1)

#%% apply transformation

I_rot,_ = interp2_bilinear(I.astype(np.float32), XI.astype(np.float32), YI.astype(np.float32))

axs2[1].imshow(I_rot)
axs2[1].set_title('transformed image')
plt.pause(0.1)

Image.fromarray((I_rot*255).astype(np.uint8)).save("im_rot.bmp")




