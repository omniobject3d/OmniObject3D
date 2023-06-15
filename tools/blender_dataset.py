import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json
import imageio
import cv2
import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)
from pytorch3d.ops import sample_farthest_points


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

def load_blender_data(basedir, half_res=False, test_ratio=0.125):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    counts = [0]
    imgs = []
    img_files = []
    poses = []

    for frame in meta['frames']:
        fname = os.path.join(basedir, 'images', frame['file_path'].split('/')[-1] + '.png')
        img_files.append(fname)
        imgs.append(cv2.imread(fname))
        pose = np.array(frame['transform_matrix'])
        pose[:,1:3] *= -1
        poses.append(pose)
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    counts.append(counts[-1] + imgs.shape[0])
    print(imgs.shape, poses.shape)

    n_images = len(imgs)
    freq_test = int(1/test_ratio)
    i_val = i_test = np.arange(0, n_images, freq_test)
    i_train = np.asarray(list(set(np.arange(n_images).tolist())-set(i_test.tolist())))
    i_split = [i_train, i_val, i_test]
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split, img_files

class Dataset:
    # Dataset Module in the format of NeuS(https://github.com/Totoro97/NeuS/blob/main/models/dataset.py).
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir') # for example: data/data_blender_processed/CATEGORY/OBJ_NAME/render

        imgs, poses, render_poses, [H, W, focal], i_split, img_files = load_blender_data(self.data_dir)

        self.fps_view = conf.get_bool('fps_view', False)
        if self.fps_view:
            self.num_views = conf.get_int('num_views', 0)
            cam_locs = poses[:,:3,-1]
            _, fps_ids = sample_farthest_points(torch.from_numpy(cam_locs[None]), None, self.num_views)
            fps_ids = fps_ids.squeeze()
            poses = np.stack([poses[i] for i in fps_ids])
            imgs = np.stack([imgs[i] for i in fps_ids])
            i_split[0] = i_split[1] = np.arange(fps_ids.shape[0])
            print('FPS image ids:', fps_ids)

        self.H, self.W, self.focal = H, W, focal
        self.image_pixels = self.H * self.W
        self.images_lis = img_files

        self.images = torch.from_numpy(imgs.astype(np.float32)[...,:3]).cpu()
        self.masks = torch.from_numpy(imgs.astype(np.float32)[...,-1:]).cpu()
        self.pose_all = torch.from_numpy(poses).to(self.device)
        self.n_images = len(self.images) # fixme: do not split

        # split
        # attention: use all the images for training for the surface reconstruction task
        i_train, i_val, i_test = i_split
        self.images_val = self.images[torch.from_numpy(i_val)]
        self.masks_val = self.masks[torch.from_numpy(i_val)]
        self.pose_all_val = self.pose_all[torch.from_numpy(i_val)]

        self.n_images_train = len(i_train)
        self.n_images_val = len(i_val)

        intrinsic = np.array([[focal, 0, H/2], [0, focal, W/2], [0, 0, 1]])
        print("intrinsics:", intrinsic)
        intrinsics = np.stack([intrinsic for _ in range(self.n_images)]).astype(np.float32)
        self.intrinsics_all = torch.from_numpy(intrinsics).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)

        self.object_bbox_min = np.asarray([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.asarray([1.01, 1.01, 1.01])

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color, mask[:, :1]], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level, set='train'):
        img_name = self.images_lis[idx] if set == 'train' else self.images_lis_val[idx]
        img = cv.imread(img_name)
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

