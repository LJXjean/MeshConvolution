# -*- coding: utf-8 -*-
"""
Extract latent vectors from test dataset.
"""

import torch
import numpy as np
import graphAE as graphAE
import graphAE_param as Param
import graphAE_dataloader as Dataloader
from plyfile import PlyData
import os

def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_indices']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
    return faces

def reconstruct_pc(param, latent_npy_fn, out_ply_folder):
    print("********** Initiate Network **********")
    model = graphAE.Model(param, test_mode=True)
    model.cuda()
    
    # Load weight
    if param.read_weight_path != "":
        print("Load " + param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.init_test_mode()
    
    model.eval()
    
    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)
    
    print("********** Get latent vectors **********", latent_npy_fn)
    latent_vectors = np.load(latent_npy_fn)
    print(latent_vectors.shape[0], "latent vectors in total.")

    reconstructed_pcs = []
    
    for n in range(latent_vectors.shape[0]):

        latent_vector = latent_vectors[n]
        print(latent_vector)
        latent_vector = torch.FloatTensor(latent_vector).cuda()
        latent_vector = latent_vector.unsqueeze(0)
        print(latent_vector)
 
        out_pcs_torch = model.reconstruct_pc(latent_vector)
        out_pcs = out_pcs_torch.cpu().numpy()
        
        out_pcs[:, :, 0:3] -= out_pcs[:, :, 0:3].mean(1).reshape((-1, 1, 3)).repeat(param.point_num, 1)
        
        reconstructed_pcs.append(out_pcs[0])

        # Dataloader.save_pc_into_ply(template_plydata, reconstruct_pc[0], out_ply_folder+"ply/"+"%08d"%(n)+"_gt.ply")
        # print(f"Reconstructed mesh {n} saved to {out_ply_folder}")
        
        
    
    reconstructed_pcs = np.array(reconstructed_pcs) 
    np.save(os.path.join(out_ply_folder, 'test_reconstructed-batch1.npy'), reconstructed_pcs)
    print(f"All reconstructed point clouds saved to {os.path.join(out_ply_folder, 'reconstructed_pcs.npy')}")




if __name__ == "__main__":
    param = Param.Parameters()
    param.read_config("../../train/0524_graphAE_capsules/30_conv_res.config")
    param.batch = 1
    param.read_weight_path = "../../train/0524_graphAE_capsules/weight_30/model_epoch0562.weight"
    
    latent_npy_fn="../../train/0524_graphAE_capsules/test_30/epoch0562/latent_vectors_batch1.npy"
    
    with torch.no_grad():
        torch.manual_seed(2)
        np.random.seed(2)

        reconstruct_pc(param, latent_npy_fn, "../../data/CAPSULES/reconstruct/output/")
