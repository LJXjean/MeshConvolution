# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
import graphAE as graphAE
import graphAE_param as Param
import graphAE_dataloader as Dataloader
from datetime import datetime
from plyfile import PlyData
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib import cm



def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    color = mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    (r,g,b) = mpl.colors.ColorConverter.to_rgb(color)
    return np.array([r,g,b])

def get_colors_from_diff_pc(diff_pc, min_error, max_error):
    colors = np.zeros((diff_pc.shape[0],3))
    mix = (diff_pc-min_error)/(max_error-min_error)
    mix = np.clip(mix, 0,1) #point_num
    cmap=cm.get_cmap('coolwarm')
    colors = cmap(mix)[:,0:3]
    return colors

def get_faces_colors_from_vertices_colors(vertices_colors, faces):
    faces_colors = vertices_colors[faces]
    faces_colors = faces_colors.mean(1)
    return faces_colors


def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_indices']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
    
    
    return faces
    

def test(param,test_npy_fn, out_ply_folder, skip_frames =0):
    
    
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param, test_mode=True)
    
    model.cuda()
    
    ##load weight
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.init_test_mode()
    
    
    model.eval()
    
    
    
    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)
    
    
    pose_sum=0
    laplace_sum=0
    test_num=0
    
    print ("**********Get test pcs**********", test_npy_fn)
    ##get ply file lst
    pc_lst= np.load(test_npy_fn)
    print (pc_lst.shape[0], "meshes in total.")
    

    geo_error_sum = 0
    laplace_error_sum=0
    mhd_error_sum=0
    pc_num = len(pc_lst)
    n = 0

    
    
    #pc_gt_original = np.array(pc_lst[0].tolist())
    #Dataloader.save_pc_into_ply(template_plydata, pc_gt_original, out_ply_folder+"%08d"%(n)+"_gt_original.ply")
    while (n<(pc_num-1)):
        
        batch = min(pc_num-n, param.batch)
        pcs = pc_lst[n:n+batch]
        height = pcs[:,:,1].mean(1)
        pcs[:,:,0:3] -= pcs[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1) ##centralize each instance

        pcs_torch = torch.FloatTensor(pcs).cuda()
        if(param.augmented_data==True):
            pcs_torch = Dataloader.get_augmented_pcs(pcs_torch)
        if(batch<param.batch):
            pcs_torch = torch.cat((pcs_torch, torch.zeros(param.batch-batch, param.point_num, 3).cuda()),0)

        out_pcs_torch,_ = model(pcs_torch)
        geo_error = model.compute_geometric_mean_euclidean_dist_error(pcs_torch[0:batch], out_pcs_torch[0:batch])
        geo_error_sum += geo_error*batch
        laplace_error_sum = laplace_error_sum + model.compute_laplace_Mean_Euclidean_Error(pcs_torch[0:batch], out_pcs_torch[0:batch])*batch
        print (n, geo_error.item())

        mhd_error = model.compute_modified_hausdorff_dist(pcs_torch[0:batch], out_pcs_torch[0:batch])
        mhd_error_sum += mhd_error*batch
        

        if(n % 128 ==0):
            print (height[0])
            pc_gt = np.array(pcs_torch[0].data.tolist()) 
            pc_gt[:,1] +=height[0]
            pc_out = np.array(out_pcs_torch[0].data.tolist())
            pc_out[:,1] +=height[0]

            diff_pc = np.sqrt(pow(pc_gt-pc_out, 2).sum(1))
            color = get_colors_from_diff_pc(diff_pc, 0, 0.02)*255
            Dataloader.save_pc_with_color_into_ply(template_plydata, pc_out, color, out_ply_folder+"%08d"%(n)+"_out.ply")
            Dataloader.save_pc_into_ply(template_plydata, pc_gt, out_ply_folder+"%08d"%(n)+"_gt.ply")

        n = n+batch


    geo_error_avg=geo_error_sum.item()/pc_num
    laplace_error_avg=  laplace_error_sum.item()/pc_num
    mhd_error_avg = mhd_error_sum.item()/pc_num

    print ("geo error:", geo_error_avg, "laplace error:", laplace_error_avg, "mhd error:", mhd_error_avg)
    

    
    
if __name__ == "__main__":
    param=Param.Parameters()
    param.read_config("../../train/0524_graphAE_capsules/30_conv_res.config")

    #param.augmented_data=True
    param.batch =32

    param.read_weight_path = "../../train/0524_graphAE_capsules/weight_30/model_epoch0562.weight"
    print (param.read_weight_path)

    test_npy_fn = "../../data/CAPSULES/test.npy"    

    out_test_folder = "../../train/0524_graphAE_capsules/test_30/epoch0562/"

    out_ply_folder = out_test_folder+"ply/"

    if not os.path.exists(out_ply_folder):
        os.makedirs(out_ply_folder)
        

    pc_lst= np.load(test_npy_fn)

    with torch.no_grad():
        torch.manual_seed(2)
        np.random.seed(2)
        
        test(param, test_npy_fn, out_ply_folder,skip_frames=0)
    

        
        
