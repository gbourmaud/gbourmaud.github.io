#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:12:44 2024

@author: guillaume
"""
import cv2 as cv
import numpy as np
from getInterpolationGridsHomography import getInterpolationGridsHomography

def stereoRectifyImages(I_A_float, I_B_float, HnA, HnB):

    #Compute bounds
    hA, wA, _ = I_A_float.shape
    corners_IA = np.array([[0, wA, wA, 0], [0, 0, hA, hA], [1, 1, 1, 1]]).T
    corners_IA_in_n = (temp := corners_IA.dot(HnA.T)) / temp[:, 2:3]
    hB, wB, _ = I_B_float.shape
    corners_IB = np.array([[0, wB, wB, 0], [0, 0, hB, hB], [1, 1, 1, 1]]).T
    corners_IB_in_n = (temp := corners_IB.dot(HnB.T)) / temp[:, 2:3]

    minxA = min(corners_IA_in_n[:,0])
    maxxA = max(corners_IA_in_n[:,0])
    minxB = min(corners_IB_in_n[:,0])
    maxxB = max(corners_IB_in_n[:,0])

    miny = min([min(corners_IA_in_n[:,1]),min(corners_IB_in_n[:,1])])
    maxy = max([max(corners_IA_in_n[:,1]),max(corners_IB_in_n[:,1])])

    w_newA = int(maxxA - minxA + 1)
    w_newB = int(maxxB - minxB + 1)
    h_new = int(maxy - miny + 1)

    H_nRA = np.array([[1., 0., minxA], [0., 1., miny], [0, 0, 1]])
    H_A_RA = np.linalg.inv(HnA) @ H_nRA

    XI_A, YI_A = getInterpolationGridsHomography(H_A_RA, h_new, w_newA)
    I_nA = cv.remap(I_A_float.astype(np.float32), XI_A.astype(np.float32), YI_A.astype(np.float32), cv.INTER_LINEAR) #very fast but inaccurate

    H_nRB = np.array([[1., 0., minxB], [0., 1., miny], [0, 0, 1]])
    H_B_RB = np.linalg.inv(HnB) @ H_nRB

    XI_B, YI_B = getInterpolationGridsHomography(H_B_RB, h_new, w_newB)
    I_nB = cv.remap(I_B_float.astype(np.float32), XI_B.astype(np.float32), YI_B.astype(np.float32), cv.INTER_LINEAR) #very fast but inaccurate
    
    return I_nA, I_nB