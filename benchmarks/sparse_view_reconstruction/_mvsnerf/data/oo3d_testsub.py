
from torch.utils.data import Dataset
from utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
import random
import json
import imageio
from scipy.spatial import distance_matrix
from .ray_utils import *
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_blender_data(basedir, half_res=False, use_testset=False):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    counts = [0]
    imgs = []
    img_files = []
    poses = []
    
    for frame in meta['frames']:
        fname = os.path.join(basedir, 'images', frame['file_path'].split('/')[-1] + '.png')
        img_files.append(fname)
        img = imageio.imread(fname)[..., :3]
        imgs.append(img)
        pose = np.array(frame['transform_matrix'])
        pose[:,1:3] *= -1
        poses.append(pose)
    imgs = np.array(imgs).astype(np.uint8) 
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if use_testset:
        with open(os.path.join(basedir, 'transforms_test.json'), 'r') as fp:
            meta_test = json.load(fp)
        render_poses = []
        test_img_files = []
        for frame in meta_test['frames']:
            pose = np.array(frame['transform_matrix'])
            pose[:, 1:3] *= -1
            render_poses.append(pose)
            test_img_files.append(frame['file_path'] + '.png')
        render_poses = torch.from_numpy(np.array(render_poses).astype(np.float32))
    else:
        render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
        test_img_files = []
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.
        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3)).astype(np.uint8) 
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).astype(np.uint8) 
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    return imgs, poses, render_poses, [H, W, focal], img_files, test_img_files


class OO3D_testsub(Dataset):
    def __init__(self, args, split='train', load_ref=False):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.args = args
        self.root_dir = os.path.dirname(args.datadir)
        self.scan = os.path.basename(args.datadir)
        self.split = split

        self.downSample = 0.5
        self.img_wh = (int(800*self.downSample),int(800*self.downSample))
        print(f'==> image down scale: {self.downSample}')

        self.pair_idx = [[40, 63, 23, 5, 45, 88, 65, 68,  8, 1, 93, 17, 50, 99, 9, 33], [78,8,20,98] ]
    
        self.scale_factor = 1.0
        self.define_transforms()
        self.near_far = [2., 6.]

        if not load_ref:
            self.read_meta()


    def define_transforms(self):
        self.transform = T.ToTensor()


    def read_source_views(self, pair_idx=None, device=torch.device("cpu")):

        src_transform = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        if pair_idx is None:
            pair_idx = self.pair_idx[0][:self.args.num_views]

        _, poses, render_poses, [H, W, focal], img_files, test_img_files = load_blender_data(os.path.join(self.root_dir, self.scan, 'render'), half_res=True, use_testset=self.args.use_testsub)

        poses = np.linalg.inv(poses)
        n_images = len(poses)
        intrinsic = np.array([[focal, 0, H/2], [0, focal, W/2], [0, 0, 1]])
        intrinsics = np.stack([intrinsic for _ in range(n_images)]).astype(np.float32)
        # multiply intrinsics and extrinsics to get projection matrix
        proj_mat_l = np.eye(4)
        proj_mat_ls  = np.stack([proj_mat_l for _ in range(n_images)]).astype(np.float32)
        proj_mat_ls[:, :3, :4] = intrinsics @ poses[:, :3, :4]

        imgs, proj_mats = [], []
        intrinsics, c2ws, w2cs = [],[],[]
        
        for i,idx in enumerate(pair_idx):
            c2w = np.linalg.inv(poses)[idx]
            w2c = poses[idx]
            c2ws.append(c2w)
            w2cs.append(w2c)
            proj_mat_l = proj_mat_ls[idx]
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]
            intrinsics.append([intrinsic.copy()])

            img_filename = img_files[idx]
            img = Image.open(img_filename)
            try:
                R, G, B , A  = img.split()
            except:
                R, G, B  = img.split()
            img = Image.merge('RGB', (R, G, B))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            imgs.append(src_transform(img))

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).squeeze(1).float().to(device)

        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, self.near_far, pose_source


    def read_meta(self):
        w, h = self.img_wh

        self.render_poses = []
        self.all_rays = []

        imgs, poses, render_poses, [H, W, focal], img_files, test_img_files = load_blender_data(os.path.join(self.root_dir, self.scan, 'render'), half_res=True, use_testset=self.args.use_testsub)

        self.src_poses = poses
        poses = np.linalg.inv(poses)
        num_src = self.args.num_views
        self.pair_idx[0][:num_src] = np.arange(num_src)
        self.test_img_files = test_img_files

        n_images = len(poses)
        intrinsic = np.array([[focal, 0, H/2], [0, focal, W/2], [0, 0, 1]])
        intrinsics = np.stack([intrinsic for _ in range(n_images)]).astype(np.float32)

        for idx in range(len(render_poses)):
            c2w = render_poses[idx]
            w2c = np.linalg.inv(render_poses)[idx]
            self.render_poses += [c2w]
            c2w = torch.FloatTensor(c2w)

            # ray directions for all pixels, same for all images (same H, W, focal)
            center = [intrinsic[0,2], intrinsic[1,2]]
            self.focal = [intrinsic[0,0], intrinsic[1,1]]
            self.directions = get_ray_directions(h, w, self.focal, center)  # (h, w, 3)
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        
            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near_far[0] * torch.ones_like(rays_o[:, :1]),
                                         self.near_far[1] * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)

        self.render_poses = np.stack(self.render_poses)
        self.all_rays = torch.stack(self.all_rays, 0) 


    def __getitem__(self, idx):
        rays = self.all_rays[idx]
        img_file = self.test_img_files[idx]
        sample = {'rays': rays,
                'img_file': img_file,
                'idx': idx}
        return sample
