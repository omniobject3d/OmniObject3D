# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import argparse
import numpy as np
import torch
import os
import random
import glob
from tqdm import tqdm
import kaolin as kal
import point_cloud_utils as pcu
import ipdb

def seed_everything(seed):
    if seed < 0:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_score(results, use_same_numer_for_test=False):
    if use_same_numer_for_test:
        results = results[:, :results.shape[0]]
    mmd = results.min(axis=1).mean()
    min_ref = results.argmin(axis=0)
    unique_idx = np.unique(min_ref)
    cov = float(len(unique_idx)) / results.shape[0]
    if mmd < 1:
        # Chamfer distance
        mmd = mmd * 1000  # for showing results
    return mmd, cov * 100


def sample_point_with_mesh_name(name, n_sample=2048, normalized_scale=1.0):
    #ipdb.set_trace()
    if '.ply' in name:
        v = pcu.load_mesh_v(name)
        point_clouds = np.random.permutation(v)[:n_sample, :]
        scale = point_clouds.max()-point_clouds.min()
        
        point_clouds = point_clouds / scale #* normalized_scale  # Make them in the same scale


        return torch.from_numpy(point_clouds).float().cuda().unsqueeze(dim=0)

    mesh_1 = kal.io.obj.import_mesh(name)

    if mesh_1.vertices.shape[0] == 0:
        return None
    vertices = mesh_1.vertices.cuda()
    scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
    mesh_v1 = vertices / scale * normalized_scale
    mesh_f1 = mesh_1.faces.cuda()
    points, _ = kal.ops.mesh.sample_points(mesh_v1.unsqueeze(dim=0), mesh_f1, n_sample)
    return points.cuda()


def chamfer_distance(ref_pcs, sample_pcs, batch_size, save_path):
    all_rec_pcs = []
    n_sample = 2048
    normalized_scale = 1.0

    save_path = os.path.join(save_path, 'covmmd')
    if os.path.exists(os.path.join(save_path,'sample.pth')):    
        all_sample_pcs=torch.load(os.path.join(save_path,'sample.pth'))
    else:    
        all_sample_pcs = []
        for name in tqdm(sample_pcs):
            # This is generated
            #ipdb.set_trace()
            all_sample_pcs.append(sample_point_with_mesh_name(name, n_sample, normalized_scale=normalized_scale))
        
        all_sample_pcs = [p for p in all_sample_pcs if p is not None]
        all_sample_pcs = torch.cat(all_sample_pcs, dim=0)
        os.makedirs(os.path.join(save_path), exist_ok=True)
        torch.save(all_sample_pcs,os.path.join(save_path,'sample.pth'))
    

    
    return 


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, save_path=None):
    chamfer_distance(ref_pcs, sample_pcs, batch_size,save_path)



def evaluate(args):
    # Set the random seed
    seed_everything(41)
    ref_path=[]


    gen_path = args.gen_path

    gen_models = glob.glob(os.path.join(gen_path, '*.obj'))
    gen_models = sorted(gen_models)
    
    gen_models = gen_models[:args.n_shape]
    with torch.no_grad():

        compute_all_metrics(gen_models, ref_path, args.batch_size, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gen_path", type=str, default='',help="path to the generated models")
    parser.add_argument("--save_path", type=str, default='./results', help="path to the save results")
    parser.add_argument("--dataset_path", type=str,default='', help="path to the original shapenet dataset")
    parser.add_argument("--n_points", type=int, default=2048, help="Number of points used for evaluation")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size to compute chamfer distance")
    parser.add_argument("--n_shape", type=int, default=7500, help="number of shapes for evaluations")
    parser.add_argument("--use_npz", type=bool, default=False, help="whether the generated shape is npz or not")
    args = parser.parse_args()
    evaluate(args)
