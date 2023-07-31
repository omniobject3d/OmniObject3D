import os
import argparse
import numpy as np
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import trimesh
from termcolor import colored
import json


USE_O3D=False
if USE_O3D:
    import open3d as o3d

def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    if USE_O3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(file, pcd)
    else:
        pcd = trimesh.points.PointCloud(vertices=points, colors=colors)
        pcd.export(file)

def getGeometryCenter(obj):
    sumWCoord = [0,0,0]
    numbVert = 0
    if obj.type == 'MESH':
        for vert in obj.vertices:
            wmtx = obj.matrix_world
            worldCoord = vert.co @ wmtx
            sumWCoord[0] += worldCoord[0]
            sumWCoord[1] += worldCoord[1]
            sumWCoord[2] += worldCoord[2]
            numbVert += 1
        sumWCoord[0] = sumWCoord[0]/numbVert
        sumWCoord[1] = sumWCoord[1]/numbVert
        sumWCoord[2] = sumWCoord[2]/numbVert
    return sumWCoord

def read_and_downsample_pcd_from_mesh(path, pbar=None, thresh=0.002, scale=1, coord_trans=False, USE_O3D=False):
    if USE_O3D:
        import open3d as o3d
        # a liiiittle bit difference in two loaders, which could be ignored temporally
        data_mesh = o3d.io.read_triangle_mesh(str(path))
        data_mesh.remove_unreferenced_vertices()
        mp.freeze_support()
        triangles = np.asarray(data_mesh.triangles)
    else:
        data_mesh = trimesh.load(path)
        try:
            data_mesh.remove_unreferenced_vertices()
        except:
            return None
        triangles = np.asarray(data_mesh.faces)

    vertices = np.asarray(data_mesh.vertices)
    vertices *= scale
    if coord_trans:
        vertices = vertices @ np.asarray([[1,0,0], [0,0,1], [0,-1,0]])
    tri_vert = vertices[triangles]
    # pbar.update(1)
    # pbar.set_description('sample pcd from', path)

    v1 = tri_vert[:, 1] - tri_vert[:, 0] # one edge vector
    v2 = tri_vert[:, 2] - tri_vert[:, 0] # another edge vector
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True) # edge length
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True) # edge length
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(sample_single_tri,
                              ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in
                               range(len(n1))), chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    # pbar.update(1)
    # pbar.set_description('random shuffle pcd index')
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    # pbar.set_description('downsample pcd')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1

    data_down = data_pcd[mask]
    
    return data_down


def eval_chamfer_distance(pred_path, gt_path, gt_scale, eval_dir, suffix='', vis=False,
    max_dist = 0.1):
    # For DTU, scale: 500, thresh: 0.2; for ours, scale: 1 thresh: 0.0004, we take 0.002 for convenience
    # max_dist: from 20 to 0.1
    thresh = 0.002 # downsample density.

    pbar = tqdm(total=9)

    pbar.set_description('read gt mesh')
    gt_down = read_and_downsample_pcd_from_mesh(gt_path, pbar, scale=gt_scale, coord_trans=True, thresh=thresh)
    if gt_down is None:
        print(colored('failed to load from {}'.format(str(gt_path)),'red'))
        return 0, 0, 0

    pbar.update(1)
    pbar.set_description('read data mesh')
    data_down = read_and_downsample_pcd_from_mesh(pred_path, pbar, thresh=thresh)
    if data_down is None:
        print(colored('failed to load from {}'.format(str(pred_path)),'red'))
        return 0, 0, 0
    trimesh.PointCloud(data_down).export("tmp.ply", "ply")

    pbar.set_description('skip data pcd masking')

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(gt_down)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    if len(dist_d2s[dist_d2s < max_dist]) == 0:
        print('no valid distance for d2s')
    pbar.update(1)
    pbar.set_description('compute stl2data')

    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(gt_down, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    if len(dist_s2d[dist_s2d < max_dist]) == 0:
        print('no valid distance for s2d')
    over_all = (mean_d2s + mean_s2d) / 2

    if vis:
        pbar.update(1)
        pbar.set_description('visualize error')
        vis_dist = 1
        R = np.array([[1, 0, 0]], dtype=np.float64)
        G = np.array([[0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 0, 1]], dtype=np.float64)
        W = np.array([[1, 1, 1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color = R * data_alpha + W * (1 - data_alpha)
        data_color[dist_d2s[:, 0] >= max_dist] = G
        stl_color = np.tile(B, (gt_down.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color = R * stl_alpha + W * (1 - stl_alpha)
        stl_color[dist_s2d[:, 0] >= max_dist] = G
        write_vis_pcd(f'{eval_dir}/vis_d2s{suffix}.ply', data_down, data_color)
        write_vis_pcd(f'{eval_dir}/vis_s2d{suffix}.ply', gt_down, stl_color)

    pbar.update(1)
    pbar.close()
    # print(" [ d2s: {:.3f} | s2d: {:.3f} | mean: {:.3f} ]".format(mean_d2s, mean_s2d, over_all))
    with open(f'{eval_dir}/result{suffix}_clip-{max_dist}.txt', 'w') as f:
        f.write(f'{mean_d2s} {mean_s2d} {over_all}')
    print('overall: {:.5f}'.format(over_all))
    return mean_d2s, mean_s2d, over_all

def fast_eval_chamfer_distance(pred_path, gt_path, eval_dir, suffix='', runtime=False, vis=False,
    max_dist = 0.1):
    # For DTU, scale: 500, thresh: 0.2; for ours, scale: 1 thresh: 0.0004, we take 0.002 for convenience
    # max_dist: from 20 to 0.1
    thresh = 0.002 # downsample density.

    pbar = tqdm(total=9)

    pbar.set_description('read gt mesh')
    gt_down = np.array(trimesh.load(gt_path).vertices)

    pbar.update(1)
    pbar.set_description('read data mesh')
    data_down = np.array(trimesh.load(pred_path).vertices)

    pbar.set_description('skip data pcd masking')

    pbar.update(1)
    pbar.set_description('compute data2stl')
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(gt_down)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_down, n_neighbors=1, return_distance=True)

    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()
    if len(dist_d2s[dist_d2s < max_dist]) == 0:
        print('no valid distance for d2s')
    pbar.update(1)
    pbar.set_description('compute stl2data')

    nn_engine.fit(data_down)
    dist_s2d, idx_s2d = nn_engine.kneighbors(gt_down, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()
    if len(dist_s2d[dist_s2d < max_dist]) == 0:
        print('no valid distance for s2d')
    over_all = (mean_d2s + mean_s2d) / 2

    if vis:
        pbar.update(1)
        pbar.set_description('visualize error')
        vis_dist = 1
        R = np.array([[1, 0, 0]], dtype=np.float64)
        G = np.array([[0, 1, 0]], dtype=np.float64)
        B = np.array([[0, 0, 1]], dtype=np.float64)
        W = np.array([[1, 1, 1]], dtype=np.float64)
        data_color = np.tile(B, (data_down.shape[0], 1))
        data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist
        data_color = R * data_alpha + W * (1 - data_alpha)
        data_color[dist_d2s[:, 0] >= max_dist] = G
        stl_color = np.tile(B, (gt_down.shape[0], 1))
        stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist
        stl_color = R * stl_alpha + W * (1 - stl_alpha)
        stl_color[dist_s2d[:, 0] >= max_dist] = G
        write_vis_pcd(f'{eval_dir}/vis_d2s{suffix}.ply', data_down, data_color)
        write_vis_pcd(f'{eval_dir}/vis_s2d{suffix}.ply', gt_down, stl_color)

    pbar.update(1)
    pbar.close()
    # print(" [ d2s: {:.3f} | s2d: {:.3f} | mean: {:.3f} ]".format(mean_d2s, mean_s2d, over_all))
    with open(f'{eval_dir}/result{suffix}_clip-{max_dist}.txt', 'w') as f:
        f.write(f'{mean_d2s} {mean_s2d} {over_all}')
    print('overall: {:.5f}'.format(over_all))
    return mean_d2s, mean_s2d, over_all


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, help='file path to the predicted surface mesh')
    parser.add_argument('--gt_path', type=str, help='file path to the ground truth raw scan')
    parser.add_argument('--transformation_path', type=str, help='file path to the transforms.json, which contains metadata for blender rendering')
    args = parser.parse_args()

    transforms_file = os.path.join(args.transformation_path)
    with open(transforms_file, 'r') as fp:
        meta = json.load(fp)
        gt_scale = meta['frames'][0]['scale']
    mean_d2s, mean_s2d, over_all = eval_chamfer_distance(args.pred_path, args.gt_path, gt_scale=gt_scale, eval_dir='./')

