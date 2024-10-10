#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:56:57 2024

@author: guillaume
"""
import numpy as np

def getInterpolationGridsHomography(Hij, h_j, w_j):

    x = np.arange(0,w_j)
    y = np.arange(0,h_j)
    X, Y = np.meshgrid(x,y) #h_j x w_j
    
    p_j = np.ones((h_j,w_j,3)) #h_j x w_j x 3
    p_j[:,:,0] = X
    p_j[:,:,1] = Y
        
    p_i = (temp := p_j @ (Hij.T)) / temp[:,:,2:3] #h_j x w_j x 3


    XI = p_i[:,:,0] #h_j x w_j
    YI = p_i[:,:,1] #h_j x w_j
 

    return XI, YI