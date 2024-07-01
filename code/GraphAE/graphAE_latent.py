# -*- coding: utf-8 -*-
"""
Extract latent vectors from test dataset.
"""

import torch
import numpy as np
import graphAE as graphAE
import graphAE_param as Param

def extract_latent_vectors(param, test_npy_fn, out_latent_npy_fn):
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
    
    print("********** Get test pcs **********", test_npy_fn)
    pc_lst = np.load(test_npy_fn)
    print(pc_lst.shape[0], "meshes in total.")
    
    latent_vectors = []
    pc_num = len(pc_lst)
    n = 0
    
    while n < pc_num:
        batch = min(pc_num - n, param.batch)
        pcs = pc_lst[n:n+batch]
        pcs[:, :, 0:3] -= pcs[:, :, 0:3].mean(1).reshape((-1, 1, 3)).repeat(param.point_num, 1)  # Centralize each instance
        
        pcs_torch = torch.FloatTensor(pcs).cuda()
        if param.augmented_data:
            pcs_torch = Dataloader.get_augmented_pcs(pcs_torch)
        if batch < param.batch:
            pcs_torch = torch.cat((pcs_torch, torch.zeros(param.batch - batch, param.point_num, 3).cuda()), 0)
        
        _, latent_vector_batch = model(pcs_torch)
        print(latent_vector_batch)
        latent_vectors.append(latent_vector_batch.cpu().numpy())
        
        n += batch
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    np.save(out_latent_npy_fn, latent_vectors)
    print(f"Saved latent vectors to {out_latent_npy_fn}")


if __name__ == "__main__":
    param = Param.Parameters()
    param.read_config("../../train/0524_graphAE_capsules/30_conv_res.config")
    param.batch = 1
    param.read_weight_path = "../../train/0524_graphAE_capsules/weight_30/model_epoch0562.weight"
    
    test_npy_fn = "../../data/CAPSULES/DatabaseCapsules.npy"
    out_latent_npy_fn = "../../train/0524_graphAE_capsules/test_30/epoch0562/latent_vectors_batch1.npy"
    
    with torch.no_grad():
        torch.manual_seed(2)
        np.random.seed(2)
        
        extract_latent_vectors(param, test_npy_fn, out_latent_npy_fn)
