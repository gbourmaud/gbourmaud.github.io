#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:01:29 2024

@author: guillaume
"""
import open3d as o3d
import numpy as np
import time

def createColoredPCD(U, colors_U):
    
    pcd = o3d.geometry.PointCloud()    
    # From numpy to Open3D
    pcd.points = o3d.utility.Vector3dVector(U)
    pcd.colors = o3d.utility.Vector3dVector(colors_U)

    return pcd

def drawCameras(K, imgs, Mpw, viz, color_first=True, show_imgs=True, cam_colors=None):
    #draw cameras
    
    
    for c in range(len(Mpw)):
        Kinv = np.linalg.inv(K[c])
        
        wIm = imgs[c].shape[1]
        hIm = imgs[c].shape[0]
        
        #show frustum
        points_in_cam_ref = [[0,0,0], 
                             [0,0,1]@(Kinv.T),
                             [wIm,0,1]@(Kinv.T), 
                             [wIm,hIm,1]@(Kinv.T),
                             [0,hIm,1]@(Kinv.T)]
        
        
        Rcw = Mpw[c][:3,:3]
        tcw = Mpw[c][:3,3]
        twc = -tcw@Rcw
        Rwc = Rcw.T
        
        points_in_w = (points_in_cam_ref @ Rcw) + twc
        lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1]
        ]
        
        if(cam_colors!=None):
            color_cur = cam_colors[c]
        elif (c==0 and color_first == True):
            color_cur = [0, 0, 0]
        else:
            colors = [1, 0, 0]
            
        colors = [color_cur for i in range(len(lines))]
            
        line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_in_w),
        lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        viz.add_geometry(line_set)
        
        #show camera coordinate system
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        frame.rotate(Rwc, np.zeros(3))
        frame.translate(twc)

        viz.add_geometry(frame)
        
        if(show_imgs == True):
            #show image
            # Define the vertices and faces for a square mesh
            #vertices = points_in_w[1:,:]
            vertices = points_in_w[[4,3,2,1],:]
            faces = np.array([
                [0, 1, 2],
                [0, 2, 3],
                [2, 1, 0],
                [3, 2, 0]
            ])
    
    
            # create the uv coordinates
            v_uv = np.array([[0, 1], [1, 1], [1, 0], 
                             [0, 1], [1, 0], [0, 0],
                             [1, 0], [1, 1], [0, 1],
                             [0, 0], [1, 0], [0, 1]])
    
            # assign the texture to the mesh
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
            mesh.textures = [o3d.geometry.Image(imgs[c].astype(np.float32))]
            mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
            mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
    
            viz.add_geometry(mesh)
    
def show3dScene(U_w, K, imgs, Mpw, color_first=True, show_imgs=True, cam_colors=None):
    
    
    WIDTH = 1280
    HEIGHT = 720
    viz = o3d.visualization.Visualizer()
    viz.create_window(width=WIDTH, height=HEIGHT)
    
    if(U_w != None):
        #get PCL colors
        p_1 = (U_w/U_w[:,2:3]) @ K[0].T
        p_1_int = p_1.astype(int)
        colors_U = (imgs[0][p_1_int[:,1],p_1_int[:,0],:])
        pcd = createColoredPCD(U_w,colors_U)
        
        #show PCL
        viz.add_geometry(pcd)
    
    #show cameras
    drawCameras(K, imgs, Mpw, viz, color_first, show_imgs, cam_colors)

    #meshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    #viz.add_geometry(meshFrame)
    
    # i=0
    # while(1):
    #     i+=1
    #     print(i)
    #     time.sleep(3)
    #     viz.poll_events()
    #     viz.update_renderer()
    viz.run()
    #viz.poll_events()
    #viz.update_renderer()
    viz.destroy_window()
    return viz

