import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
import json
import logging
from util import get_image_to_tensor_balanced, get_mask_to_tensor

from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

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

class OO3DDataset(torch.utils.data.Dataset):
    """
    Dataset from DVR (Niemeyer et al. 2020)
    Provides 3D-R2N2 and NMR renderings
    """

    def __init__(
        self,
        path,
        stage="train",
        list_prefix="softras_",
        image_size=None,
        sub_format="shapenet",
        scale_focal=True,
        max_imgs=100000,
        z_near=1.2,
        z_far=4.0,
        skip_step=None,
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        self.base_path = path
        assert os.path.exists(self.base_path)
        all_objs = []

        if stage == 'train':
            cats = sorted(os.listdir(self.base_path))  # default: train on all the categories
            # cats = ['toy_train', 'bread', 'cake', 'toy_boat', 'hot_dog', 'wallet', 'pitaya', 'squash', 'handbag', 'apple'] # change cats for individual needs
            for cat in cats:
                scans = sorted(os.listdir(os.path.join(self.base_path, cat)))
                scans = scans[3:]
                objs = [(cat, os.path.join(path, cat, x)) for x in scans]
                all_objs.extend(objs)
        if stage == 'val':
            cats = sorted(os.listdir(self.base_path)) # default: train on all the categories
            # cats = ['toy_train', 'bread', 'cake', 'toy_boat', 'hot_dog', 'wallet', 'pitaya', 'squash', 'handbag', 'apple'] # change cats for individual needs
            for cat in cats:
                scans = os.listdir(os.path.join(self.base_path, cat))
                scans.sort()
                scans = scans[:3]
                objs = [(cat, os.path.join(path, cat, x)) for x in scans]
                all_objs.extend(objs)
        elif stage == 'test':
            cat = os.path.basename(self.base_path)
            scans = os.listdir(os.path.join(self.base_path))
            scans.sort()
            objs = [(cat, os.path.join(path, x)) for x in scans]
            all_objs.extend(objs)
        elif stage == 'testsub':
            scans = sorted(os.listdir(self.base_path))
            objs = [('category_agnostic', os.path.join(path, x)) for x in scans]
            all_objs.extend(objs)

        self.all_objs = all_objs
        self.stage = stage

        self._coord_trans = torch.diag(
            torch.tensor([1, -1, -1, 1], dtype=torch.float32)
        )

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print(
            "Loading OO3D dataset",
            self.base_path,
            "stage",
            stage,
            len(self.all_objs),
            "objs",
        )
    
        self.image_size = image_size
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs
    
        self.z_near = z_near
        self.z_far = z_far

        self.lindisp = False
    
    def __len__(self):
        return len(self.all_objs)
    
    def __getitem__(self, index):
    
        cat, data_dir = self.all_objs[index]
        data_dir = os.path.join(data_dir, 'render')
        imgs, poses, render_poses, [H, W, focal], img_files, test_img_files = load_blender_data(data_dir, half_res=True, use_testset=self.stage=='testsub')
        bboxes = []
        masks = []
        images = torch.zeros((imgs.shape[0], 3, H, W))
        for i, img in enumerate(imgs):
            images[i] = self.image_to_tensor(img) 
            mask = (img != 0).all(axis=-1)[..., None].astype(np.uint8) * 255
            mask_tensor = self.mask_to_tensor(mask)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                rmin, rmax = [0, H-1]
            else:
                rmin, rmax = rnz[[0, -1]]
            if len(cnz) == 0:
                cmin, cmax = [0, W-1]
            else:
                cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
            bboxes.append(bbox)
            masks.append(mask_tensor)

        bboxes = torch.stack(bboxes)
        masks = torch.stack(masks)
        poses_origin = poses
        poses = torch.from_numpy(poses.astype(np.float32)).float() @ self._coord_trans
        render_poses = render_poses.float() @ self._coord_trans
        n_images = len(images) # fixme: do not split
    
        intrinsic = np.array([[focal, 0, H/2], [0, focal, W/2], [0, 0, 1]])
        intrinsics = np.stack([intrinsic for _ in range(n_images)]).astype(np.float32)
        intrinsics_all = torch.from_numpy(intrinsics)
        intrinsics_all_inv = torch.inverse(intrinsics_all)
    
        focal = torch.tensor((focal, focal), dtype=torch.float32)
        c = torch.tensor((H/2, W/2), dtype=torch.float32)
    
        result = {
            "path": data_dir,
            "img_id": index,
            "focal": focal,
            "images": images,
            "masks": masks,
            "poses": poses,
            "poses_origin" : poses_origin,
            "c": c,
            "bbox": bboxes,
            "render_poses": render_poses,
            "test_img_files": test_img_files,
        }
        return result
