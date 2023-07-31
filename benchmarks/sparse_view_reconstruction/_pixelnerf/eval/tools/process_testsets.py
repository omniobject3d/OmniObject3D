import os
import json
import shutil
import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points
from mesh_eval import read_and_downsample_pcd_from_mesh
import trimesh

root = '/mnt/petrelfs/wutong/DATA/OO3D/First_release/OO3D_0717_rename/FINAL/test/OO3D_renders/'
scans_root = '/mnt/petrelfs/wutong/DATA/OO3D/First_release/OO3D_0717_rename/FINAL/test/OO3D_scans'
dst = '/mnt/petrelfs/wutong/OO3D_Benchmarks_Codebase/sparse_nerf/data/'
# scans_dst = '/mnt/petrelfs/wutong/OO3D_Benchmarks_Codebase/sparse_nerf/data/gt_pcds'
scans_dst = '/mnt/petrelfs/wutong/OO3D_Benchmarks_Codebase/sparse_nerf/data/gt_pcds_low'
def process_testset(basedir, dstdir, obj_name, given_views=3, eval_views=10):

    num_views = given_views + eval_views

    with open(os.path.join(basedir, 'render/transforms.json'), 'r') as fp:
        meta = json.load(fp)

    img_files = []
    poses = []
    for frame in meta['frames']:
        fname = os.path.join(basedir, 'render/images', frame['file_path'].split('/')[-1] + '.png')
        img_files.append(fname)
        pose = np.array(frame['transform_matrix'])
        pose[:, 1:3] *= -1
        poses.append(pose)

    poses = np.array(poses).astype(np.float32)

    cam_locs = poses[:, :3, -1]
    _, fps_ids = sample_farthest_points(torch.from_numpy(cam_locs[None]), None, num_views)
    fps_ids = fps_ids.squeeze()
    print('FPS image ids:', fps_ids)

    train_dir = os.path.join(dstdir, 'train', obj_name)
    test_dir = os.path.join(dstdir, 'test', obj_name)
    os.makedirs(os.path.join(train_dir, 'render', 'images'))
    os.makedirs(os.path.join(test_dir, 'render', 'images'))
    os.makedirs(os.path.join(test_dir, 'render', 'depths'))
    train_meta = dict(camera_angle_x=meta['camera_angle_x'], frames=[])
    test_meta = dict(camera_angle_x=meta['camera_angle_x'], frames=[])
    for id in fps_ids[:given_views]:
        shutil.copy(os.path.join(basedir, 'render/images/r_{}.png'.format(id)), os.path.join(train_dir, 'render/images/r_{}.png'.format(id)))
        train_meta['frames'].append(meta['frames'][id])
    for id in fps_ids[given_views:]:
        shutil.copy(os.path.join(basedir, 'render/images/r_{}.png'.format(id)), os.path.join(test_dir, 'render/images/r_{}.png'.format(id)))
        shutil.copy(os.path.join(basedir, 'render/depths/r_{}_depth.exr'.format(id)), os.path.join(test_dir, 'render/depths/r_{}_depth.exr'.format(id)))
        test_meta['frames'].append(meta['frames'][id])
    with open(os.path.join(train_dir, 'render/transforms.json'), 'w') as fp:
        json.dump(train_meta, fp, indent=4)
    with open(os.path.join(train_dir, 'render/transforms_test.json'), 'w') as fp:
        json.dump(test_meta, fp, indent=4)
    with open(os.path.join(test_dir, 'render/transforms.json'), 'w') as fp:
        json.dump(test_meta, fp, indent=4)

def extract_gt_pcds(mesh_file, transforms_file, dstdir):
    with open(transforms_file, 'r') as fp:
        meta = json.load(fp)
        gt_scale = meta['frames'][0]['scale']
    gt_down = read_and_downsample_pcd_from_mesh(mesh_file, scale=gt_scale, coord_trans=True, thresh=0.01)
    os.makedirs(dstdir, exist_ok=True)
    trimesh.PointCloud(gt_down).export(os.path.join(dstdir, 'pcd.ply'))

n = 0
cats = sorted(os.listdir(root))
for cat in cats:
    objs = sorted(os.listdir(os.path.join(root, cat)))
    for obj in objs:
        # process_testset(os.path.join(root, cat, obj), dst, 'obj_{:03d}'.format(n))

        if os.path.exists(os.path.join(scans_dst,'obj_{:03d}'.format(n))):
            n += 1
            continue
        try:
            print(cat, obj, n, 'starts')
            extract_gt_pcds(mesh_file=os.path.join(scans_root, cat, obj, 'Scan/Scan.obj'),
                            transforms_file=os.path.join(root, cat, obj, 'render/transforms.json'),
                            dstdir=os.path.join(scans_dst,'obj_{:03d}'.format(n)))
        except:
            print(cat, obj, 'failed')
        n += 1



