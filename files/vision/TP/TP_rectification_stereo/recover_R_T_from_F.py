#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:44:54 2024

@author: guillaume
"""
import cv2 as cv

def recover_R_T_from_F(K, F_AB, p_A, p_B):

    E_AB =  K.T @ F_AB @ K
    nInliersChirality, R_BA, t_BA, maskChirality = cv.recoverPose(	E_AB.T, p_A[:,:2], p_B[:,:2], K)
    print(R_BA)
    print(t_BA)
    
    return R_BA, t_BA.ravel(), (maskChirality!=0).ravel()
