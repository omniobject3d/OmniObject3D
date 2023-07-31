
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
import tqdm
from scipy.spatial import distance_matrix
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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

def load_blender_data(basedir, downSample=0.5, test_ratio=0.125):
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

    n_images = len(poses)
    freq_test = int(1/test_ratio)

    H, W = imgs[0].shape[:2]

    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    H = H * downSample
    W = W * downSample
    focal = focal * downSample

    return poses, render_poses, [H, W, focal]

class OO3DDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        """
        img_wh should be set to a tuple ex: (1152, 864) to enable test mode!
        """
        self.root_dir = args.datadir
        self.args = args
        self.mode = mode
        
        assert self.mode in ['train', 'val', 'test'], \
            'mode must be either "train", "val" or "test"!'
        self.img_wh = [800., 800.]
        self.downSample = 0.5
        self.scale_factor = 1.0
        self.near_far = [2., 6.]
        self.max_len = -1.
        self.n_views = 3
        self.len_pair = 30
        self.build_proj_mats()
        self.build_metas()
        self.define_transforms()
        print(f'==> image down scale: {self.downSample}')
    
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                                    #             std=[0.229, 0.224, 0.225]),
                                    ])

    def build_metas(self):
        self.metas = []
        self.id_list = []
        for scan in self.scans:
            scan_idx = self.scans.index(scan)
            num_viewpoint = 100
            for ref_view in range(num_viewpoint):
                # src_views = random.sample(range(num_viewpoint), 10)
                src_views = self.ref_src_pairs[scan_idx][ref_view]
                self.metas += [(scan, ref_view, src_views)]
                self.id_list.append([ref_view] + src_views)
        self.id_list = np.unique(self.id_list)

    def build_proj_mats(self):

        # change cats to suit individual needs
        self.cats = sorted(os.listdir(self.root_dir))  # all cats for training in the challenge
        # self.cats = ['toy_train', 'bread', 'cake', 'toy_boat', 'hot_dog', 'wallet', 'pitaya', 'squash', 'handbag', 'apple'] # sub cats for evaluation in the paper
        
        self.scans = []
        proj_mats, intrinsics_all, world2cams, cam2worlds, ref_src_pairs_all = [], [], [], [], []
        if os.path.exists('ref_src_pairs.json'):
            self.ref_src_pairs_dic = json.load(open('ref_src_pairs.json',"r"))
            offline_loading = True
            self.poses_dic = json.load(open('poses.json',"r"))
        else:
            offline_loading = False

        for cat in tqdm.tqdm(self.cats):
            scans = os.listdir(os.path.join(self.root_dir, cat))
            scans.sort()
            if self.mode == 'train':
                scans = scans[3:]
            else:
                scans = scans[:3]
            for scan in scans:
                self.scans.append(scan)
                
                if not offline_loading:
                # online loading
                    poses, render_poses, [H, W, focal] = load_blender_data(os.path.join(self.root_dir, cat, scan, 'render'), downSample=self.downSample)
                    ref_src_pairs = self.find_nearest_pairs(poses) 
                else:
                # offline loading
                    H = 400.
                    W = 400.
                    focal = 555.5555155968841
                    poses = np.concatenate((np.array(self.poses_dic[scan]), np.repeat(np.array([0.,0.,0.,1.])[None,None], 100, axis=0)),axis=1)
                    ref_src_pairs = self.ref_src_pairs_dic[scan]    

                poses = np.linalg.inv(poses)
                n_images = len(poses)
                intrinsic = np.array([[focal, 0, H/2], [0, focal, W/2], [0, 0, 1]])
                intrinsics = np.stack([intrinsic for _ in range(n_images)]).astype(np.float32)
                # multiply intrinsics and extrinsics to get projection matrix
                proj_mat_l = np.eye(4)
                proj_mat_l  = np.stack([proj_mat_l for _ in range(n_images)]).astype(np.float32)
                proj_mat_l[:, :3, :4] = intrinsics @ poses[:, :3, :4]
                proj_mats += [(proj_mat_l, self.near_far)]
                world2cams += [poses]
                cam2worlds += [np.linalg.inv(poses)]
                ref_src_pairs_all += [ref_src_pairs]
                intrinsics_all += [intrinsics]
        
        self.ref_src_pairs = np.stack(ref_src_pairs_all)
        self.proj_mats, self.intrinsics = np.stack(proj_mats), np.stack(intrinsics_all)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
    

    def read_depth(self, filename):
        try:
            depth_h = cv2.imread(filename ,cv2.IMREAD_UNCHANGED)[...,-1]
        except:
            depth_h = np.zeros((800, 800)).astype(np.float32)
        depth_h[depth_h==65504.] = 0.
        depth = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample, interpolation=cv2.INTER_NEAREST)  
        mask = depth > 0
        return depth, mask


    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def find_nearest_pairs(self, poses):
        xyzs = poses[:,:3,-1]
        distances = distance_matrix(xyzs, xyzs, p=2)
        ref_src_pairs = []
        for dist in distances:
            rank = np.argsort(dist)[:self.len_pair]
            xyzs = poses[:,:3,-1][rank]
            centroids = farthest_point_sample(torch.from_numpy(xyzs[None]), self.len_pair)
            src_views = [ rank[centroids[0][i].item()] for i in range(self.len_pair)]
            ref_src_pairs.append(src_views)
        return ref_src_pairs

    def __getitem__(self, idx):
        sample = {}
        scan, target_view, src_views = self.metas[idx]
        scan_idx = self.scans.index(scan)
        num_src_views = self.args.num_source_views
        if self.mode=='train':
            ids = torch.randperm(self.len_pair)[:num_src_views]
            view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            view_ids = [src_views[i] for i in range(num_src_views)] + [target_view]

        affine_mat, affine_mat_inv = [], []
        imgs, depths_h, bboxes = [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.root_dir, scan[:-4], scan, f'render/images/r_{vid}.png')
            depth_filename = os.path.join(self.root_dir, scan[:-4], scan, f'render/depths/r_{vid}_depth.exr')

            img = Image.open(img_filename)
            try:
                R, G, B ,A  = img.split()
            except:
                R, G, B   = img.split()
            img = Image.merge('RGB', (R, G, B))
            img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
            img = img.resize(img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]
            
            proj_mat_ls, near_far = self.proj_mats[scan_idx]
            proj_mat_ls  = proj_mat_ls[vid]
            
            intrinsics.append(self.intrinsics[scan_idx][vid])
            w2cs.append(self.world2cams[scan_idx][vid])
            c2ws.append(self.cam2worlds[scan_idx][vid])
            
            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]

            if os.path.exists(depth_filename):
                depth_h, mask = self.read_depth(depth_filename)
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((1, 1)))

            mask = (depth_h != 0).astype(np.uint8) * 255
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                rmin, rmax = [0, img_wh[1]-1]
            else:
                rmin, rmax = rnz[[0, -1]]
            if len(cnz) == 0:
                cmin, cmax = [0, img_wh[0]-1]
            else:
                cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)
            bboxes.append(bbox)
            near_fars.append(near_far)
        
        imgs = torch.stack(imgs).float()
        depths_h = np.stack(depths_h)
        bboxes = torch.stack(bboxes)
        proj_mats = np.stack(proj_mats)[:, :3]
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)

        intrinsics_4 = np.zeros((intrinsics.shape[0],4,4), dtype=np.float32)
        intrinsics_4[:,:3,:3] = intrinsics
        intrinsics_4[:,3,3] = 1.

        rgb = imgs[-1].permute(1,2,0)
        rgb_file = os.path.join(self.root_dir, scan[:-4], scan, f'render/images/r_{view_ids[-1]}.png')
        camera = np.concatenate(([img_wh[0], img_wh[1]], intrinsics_4[-1].flatten(), c2ws[-1].flatten())).astype(np.float32)
        camera = torch.from_numpy(camera)

        src_rgbs = imgs[:-1].permute(0,2,3,1)
        src_cameras = [np.concatenate(([img_wh[0], img_wh[1]], intrinsics_4[i].flatten(), c2ws[i].flatten())).astype(np.float32) for i in range(num_src_views)]
        src_cameras = torch.from_numpy(np.stack(src_cameras, axis=0))

        depth_range = torch.Tensor(self.near_far)

        return {'rgb': rgb,
                'camera': camera,
                'rgb_path': rgb_file,
                'src_rgbs': src_rgbs,
                'src_cameras': src_cameras,
                'depth_range': depth_range,
                }
