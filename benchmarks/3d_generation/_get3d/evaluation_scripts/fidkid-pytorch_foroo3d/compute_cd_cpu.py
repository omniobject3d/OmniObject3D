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
from pytorch3d.loss import chamfer_distance


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

def batch_chamfer_distance(ref_pcs, sample_pcs, batch_size,save_name):
    all_rec_pcs = []
    n_sample = 2048
    normalized_scale = 1.0
    #ipdb.set_trace()
    if os.path.exists(os.path.join(ref_pcs,'covmmd','gt.pth')):
        all_rec_pcs=torch.load(os.path.join(ref_pcs,'covmmd','gt.pth'), map_location=torch.device('cpu'))
    else:
        print('no gt')


    if os.path.exists(os.path.join(sample_pcs,'covmmd','sample.pth')):
        all_sample_pcs=torch.load(os.path.join(sample_pcs,'covmmd','sample.pth'), map_location=torch.device('cpu'))
    else:
        print('no sample')

    all_cd = []
    for i_ref_p in tqdm(range(len(all_rec_pcs))):
        ref_p = all_rec_pcs[i_ref_p]
        cd_lst = []
        for sample_b_start in range(0, len(all_sample_pcs), batch_size):
            sample_b_end = min(len(all_sample_pcs), sample_b_start + batch_size)
            sample_batch = all_sample_pcs[sample_b_start:sample_b_end]

            batch_size_sample = sample_batch.size(0)
            # chamfer = kal.metrics.pointcloud.chamfer_distance(
            #     ref_p.unsqueeze(dim=0).expand(batch_size_sample, -1, -1),
            #     sample_batch)
            chamfer = chamfer_distance(
                ref_p.unsqueeze(dim=0).expand(batch_size_sample, -1, -1),
                sample_batch)[0]
            cd_lst.append(chamfer)
        cd_lst = torch.cat(cd_lst, dim=0)
        all_cd.append(cd_lst.unsqueeze(dim=0))
    all_cd = torch.cat(all_cd, dim=0)
    return all_cd


def compute_all_metrics(genpath, gtpath, batch_size, save_name=None):
    results = batch_chamfer_distance(gtpath, genpath, batch_size,save_name).data.cpu().numpy()
    #ipdb.set_trace()
    #results = results[:, :results.shape[0] * 5]  # Generation is 5 time of the testing set
    cd_mmd, cd_cov = get_score(results, use_same_numer_for_test=False)
    #ipdb.set_trace()
    print( 'cov,mmd:',(cd_cov, cd_mmd))



def evaluate(args):
    # Set the random seed
    seed_everything(41)

    if not os.path.exists(os.path.join(args.gt_path,'covmmd','gt.pth')):
        print('error')

    with torch.no_grad():
        #ipdb.set_trace()
        compute_all_metrics(args.gen_path, args.gt_path, args.batch_size, args.save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, default='./results/method_name/', help="path to the save results")
    parser.add_argument("--gt_path", type=str,default='', help="path to the original shapenet dataset")
    parser.add_argument("--gen_path", type=str, default='',help="path to the generated models")
    parser.add_argument("--n_points", type=int, default=2048, help="Number of points used for evaluation")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size to compute chamfer distance")
    parser.add_argument("--n_shape", type=int, default=7500, help="number of shapes for evaluations")
    parser.add_argument("--use_npz", type=bool, default=False, help="whether the generated shape is npz or not")
    args = parser.parse_args()
    evaluate(args)
