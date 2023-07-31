
from torch.utils.data import Dataset
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
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from .ray_utils import get_ray_directions, get_rays

def farthest_point_sample(xyz, npoint): 

    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     
    distance = torch.ones(B, N).to(device) * 1e10                       

    batch_indices = torch.arange(B, dtype=torch.long).to(device)       
    
    barycenter = torch.sum((xyz), 1)                                    
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist,1)[1]                                     

    for i in range(npoint):
        # print("-------------------------------------------------------")
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest                                      
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)        
        dist = torch.sum((xyz - centroid) ** 2, -1)                     
        mask = dist < distance
        distance[mask] = dist[mask]                                    
        farthest = torch.max(distance, -1)[1]                        
    
    return centroids

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

def load_blender_data(basedir, downSample=1.0, test_ratio=0.125):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    poses = []
    for frame in meta['frames']:
        pose = np.array(frame['transform_matrix'])
        pose[:,1:3] *= -1
        poses.append(pose)
    poses = np.array(poses).astype(np.float32)

    n_images = len(poses)
    freq_test = int(1/test_ratio)

    H, W = 800, 800
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,80+1)[:-1]], 0)
    H = H * downSample
    W = W * downSample
    focal = focal * downSample

    return poses, render_poses, [H, W, focal]

class OO3D_ft_Dataset(Dataset):
    def __init__(self, args, mode, load_ref=False):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.args = args
        self.root_dir = os.path.dirname(args.datadir)
        self.scan = os.path.basename(args.datadir)
        self.mode = mode

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

        if pair_idx is None:
            pair_idx = self.pair_idx[0][:self.args.num_source_views]

        poses, render_poses, [H, W, focal] = load_blender_data(os.path.join(self.root_dir, self.scan, 'render'), downSample=self.downSample)
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

            img_filename = os.path.join(self.root_dir, self.scan, f'render/images/r_{idx}.png')
            img = Image.open(img_filename)
            try:
                R, G, B , A  = img.split()
            except:
                R, G, B  = img.split()
            img = Image.merge('RGB', (R, G, B))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)
            imgs.append(img)

        pose_source = {}
        pose_source['c2ws'] = torch.from_numpy(np.stack(c2ws)).float().to(device)
        pose_source['w2cs'] = torch.from_numpy(np.stack(w2cs)).float().to(device)
        pose_source['intrinsics'] = torch.from_numpy(np.stack(intrinsics)).squeeze(1).float().to(device)

        imgs = torch.stack(imgs).float().unsqueeze(0).to(device)
        proj_mats = torch.from_numpy(np.stack(proj_mats)[:,:3]).float().unsqueeze(0).to(device)
        return imgs, proj_mats, self.near_far, pose_source


    def read_depth(self, filename):
        depth_h = cv2.imread(filename ,cv2.IMREAD_UNCHANGED)[...,-1]
        depth_h[depth_h==65504.] = 0.
        depth = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample, interpolation=cv2.INTER_NEAREST)  
        mask = depth > 0
        return depth


    def read_meta(self):
        w, h = self.img_wh

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_depth = []

        poses, render_poses, [H, W, focal] = load_blender_data(os.path.join(self.root_dir, self.scan, 'render'), downSample=self.downSample)
        self.poses_all = poses
        poses = np.linalg.inv(poses)
        # select source views
        xyzs = self.poses_all[:,:3,-1]
        distances = distance_matrix(xyzs, xyzs, p=2)
        rank = np.argsort(distances[50])
        num_select = 30
        print(f'==> len pair: {num_select}')

        xyzs = xyzs[rank[:num_select]]
        rank = rank[:num_select]
        num_src = self.args.num_source_views
        centroids = farthest_point_sample(torch.from_numpy(xyzs[None]), num_src)

        src_pair_idx = [ rank[centroids[0][i].item()] for i in range(num_src)]
        self.pair_idx[0][:num_src] = src_pair_idx

        # select reference views
        xyzs = self.poses_all[:,:3,-1]
        distances = distance_matrix(xyzs, xyzs, p=2)
        rank = np.argsort(distances[50])
        centroids = farthest_point_sample(torch.from_numpy(xyzs[None]), 100)
        ref_pair_idx = [ centroids[0][i].item() for i in range(100)]
        for i in range(num_src):
            ref_pair_idx.remove(src_pair_idx[i])
        
        self.pair_idx[0][num_src:] = ref_pair_idx[:16-num_src]
        self.pair_idx[1] =  ref_pair_idx[16-num_src : 26-num_src]

        self.img_idx = self.pair_idx[0] if 'train' == self.mode else self.pair_idx[1]
        print(f'===> {self.mode}ing index: {self.img_idx}')

        n_images = len(poses)
        intrinsic = np.array([[focal, 0, H/2], [0, focal, W/2], [0, 0, 1]])
        intrinsics = np.stack([intrinsic for _ in range(n_images)]).astype(np.float32)

        intrinsics_4 = np.zeros((intrinsics.shape[0],4,4), dtype=np.float32)
        intrinsics_4[:,:3,:3] = intrinsics
        intrinsics_4[:,3,3] = 1.

        self.intrinsics_4 = intrinsics_4

        # multiply intrinsics and extrinsics to get projection matrix
        proj_mat_l = np.eye(4)
        proj_mat_l  = np.stack([proj_mat_l for _ in range(n_images)]).astype(np.float32)
        proj_mat_l[:, :3, :4] = intrinsics @ poses[:, :3, :4]
 
        for idx in self.img_idx:
            c2w = np.linalg.inv(poses)[idx]
            w2c = poses[idx]
            self.poses += [c2w]
            c2w = torch.FloatTensor(c2w)

            image_filename = os.path.join(self.root_dir, self.scan, f'render/images/r_{idx}.png')
            depth_filename = os.path.join(self.root_dir, self.scan, f'render/depths/r_{idx}_depth.exr')
            self.image_paths += [image_filename]

            img = Image.open(image_filename)
            try:
                R, G, B , A  = img.split()
            except:
                R, G, B  = img.split()
            img = Image.merge('RGB', (R, G, B))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGBA
            self.all_rgbs += [img]

            if os.path.exists(depth_filename) and self.mode!='train':
                depth = self.read_depth(depth_filename)
                depth *= self.scale_factor
                self.all_depth += [torch.from_numpy(depth).float().view(-1,1)]
        
            # ray directions for all pixels, same for all images (same H, W, focal)
            center = [intrinsic[0,2], intrinsic[1,2]]
            self.focal = [intrinsic[0,0], intrinsic[1,1]]
            self.directions = get_ray_directions(h, w, self.focal, center)  # (h, w, 3)
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        
            self.all_rays += [torch.cat([rays_o, rays_d,
                                         self.near_far[0] * torch.ones_like(rays_o[:, :1]),
                                         self.near_far[1] * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)

        self.all_spiral_rays = []
        self.render_poses = render_poses
        for idx in range(len(render_poses)):
            rays_o, rays_d = get_rays(self.directions, render_poses[idx] @ torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32)))  # both (h*w, 3)
            self.all_spiral_rays += [torch.cat([rays_o, rays_d,
                                         self.near_far[0] * torch.ones_like(rays_o[:, :1]),
                                         self.near_far[1] * torch.ones_like(rays_o[:, :1])],
                                        1)]  # (h*w, 8)

        src_imgs = []
        for idx in self.pair_idx[0][:3]:
            image_filename = os.path.join(self.root_dir, self.scan, f'render/images/r_{idx}.png')
            depth_filename = os.path.join(self.root_dir, self.scan, f'render/depths/r_{idx}_depth.exr')
            self.image_paths += [image_filename]

            img = Image.open(image_filename)
            try:
                R, G, B , A  = img.split()
            except:
                R, G, B  = img.split()
            img = Image.merge('RGB', (R, G, B))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            img = img.permute(1,2,0)
            src_imgs += [img]

        src_cameras = [np.concatenate(([self.img_wh[0], self.img_wh[1]], self.intrinsics_4[i].flatten(), self.poses_all[i].flatten())).astype(np.float32) for i in self.pair_idx[0][:3]]
        self.src_cameras = torch.from_numpy(np.stack(src_cameras, axis=0))
        self.src_imgs = torch.stack(src_imgs).float()

        self.poses = np.stack(self.poses)

        if 'train' == self.mode:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_spiral_rays = torch.stack(self.all_spiral_rays, 0) 
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_depth = torch.stack(self.all_depth, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def __len__(self):
        if self.mode == 'train':
            return len(self.all_rgbs)
        return len(self.all_rgbs)


    def __getitem__(self, idx):
        if self.mode == 'train':  # use data in the buffers
            rgb = self.all_rgbs[idx]
            rgb_file = os.path.join(self.root_dir, self.scan, f'render/images/r_{self.pair_idx[0][idx]}.png')
            camera = np.concatenate(([self.img_wh[0], self.img_wh[1]], self.intrinsics_4[self.pair_idx[0][idx]].flatten(), self.poses_all[self.pair_idx[0][idx]].flatten())).astype(np.float32)
            camera = torch.from_numpy(camera)
            src_rgbs = self.src_imgs
            src_cameras = torch.from_numpy(np.stack(self.src_cameras, axis=0))
            depth_range = torch.Tensor(self.near_far)

            return {'rgb': rgb,
                    'camera': camera,
                    'rgb_path': rgb_file,
                    'src_rgbs': src_rgbs,
                    'src_cameras': src_cameras,
                    'depth_range': depth_range,
                    }

        else:  # create data for each image separately
            rgb = self.all_rgbs[idx]
            rgb_file = os.path.join(self.root_dir, self.scan, f'render/images/r_{self.pair_idx[1][idx]}.png')
            camera = np.concatenate(([self.img_wh[0], self.img_wh[1]], self.intrinsics_4[self.pair_idx[1][idx]].flatten(), self.poses_all[self.pair_idx[1][idx]].flatten())).astype(np.float32)
            camera = torch.from_numpy(camera)

            src_rgbs = self.src_imgs
            src_cameras = self.src_cameras
            depth = self.all_depth[idx]
            depth_range = torch.Tensor(self.near_far)

            return {'rgb': rgb,
                    'camera': camera,
                    'rgb_path': rgb_file,
                    'src_rgbs': src_rgbs,
                    'src_cameras': src_cameras,
                    'depth': depth,
                    'depth_range': depth_range,
                    }
