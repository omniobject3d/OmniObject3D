from opt import config_parser
from torch.utils.data import DataLoader

from data import dataset_dict

# models
from models import *
from renderer import *
from utils import *
from data.ray_utils import ray_marcher,ray_marcher_fine

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg')
import pytorch_msssim
import torch
import numpy as np
import cv2
import imageio

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_mesh(resolution=512, scene='', savedir='', threshold=0.3, scale_mats_np=None, **kwargs):
    import trimesh
    os.makedirs(savedir, exist_ok=True)
    vertices, triangles = extract_geometry(resolution=resolution, threshold=threshold, **kwargs)
    if scale_mats_np is not None:
        vertices = vertices * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(savedir, "{0}".format(scene)+'.ply')
    mesh.export(mesh_path)
    print("mesh saved at " + mesh_path)

def extract_geometry(network_fn, network_query_fn, resolution=128, threshold=0.001, **kwargs):
    bound_min = torch.tensor([-1., -1., -1.]).cuda() * 0.85
    bound_max = torch.tensor([1, 1, 1]).cuda() * 0.85
    def query_func(pts):
        pts = pts.view(512,512,3).cuda()
        viewdirs = torch.zeros_like(pts).cuda()
        H, W = system.imgs.shape[-2:]
        inv_scale = torch.tensor([W - 1, H - 1]).cuda()
        w2c_ref, intrinsic_ref = system.pose_source['w2cs'][0].clone(), system.pose_source['intrinsics'][0].clone()
        intrinsic_ref[:2] *= args.imgScale_test/args.imgScale_train
        pts_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, pts, inv_scale, near=system.near_far_source[0], far=system.near_far_source[1], pad=args.pad*args.imgScale_test, lindisp=args.use_disp)
        input_feat = gen_pts_feats(system.imgs, system.volume, pts,  system.pose_source, pts_NDC, args.feat_dim, None, args.img_downscale, args.use_color_volume, args.net_type)
        raw = network_query_fn(pts_NDC, viewdirs, input_feat, system.render_kwargs_train['network_fn']).view(-1,4)
        return raw[...,-1]

    return extract_geometry_(bound_min,
                            bound_max,
                            resolution=resolution,
                            threshold=threshold,
                            query_func=query_func)

def extract_geometry_(bound_min, bound_max, resolution, threshold, query_func, N = 64):
    import mcubes
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func, N)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def extract_fields(bound_min, bound_max, resolution, query_func, N = 64):
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.args.dir_dim = 3
        self.idx = 0

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']
        self.render_kwargs_train.pop('network_mvs')

        dataset = dataset_dict[self.args.dataset_name]
        self.train_dataset = dataset(args, split='train')
        self.val_dataset   = dataset(args, split='val')
        self.init_volume()
        self.grad_vars += list(self.volume.parameters())

    def decode_batch(self, batch):
        rays = batch['rays'].squeeze()  # (B, 8)
        rgbs = batch['rgbs'].squeeze()  # (B, 3)
        return rays, rgbs

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
        return (data - mean) / std

    def init_volume(self):
        self.imgs, self.proj_mats, self.near_far_source, self.pose_source = self.train_dataset.read_source_views(device=device)
        ckpts = torch.load(args.ckpt)
        if 'volume' not in ckpts.keys():
            self.MVSNet.train()
            with torch.no_grad():
                volume_feature, _, _ = self.MVSNet(self.imgs, self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
        else:
            volume_feature = ckpts['volume']['feat_volume']
            print('load ckpt volume.')

        self.imgs = self.unpreprocess(self.imgs)
        self.density_volume = None
        self.volume = RefVolume(volume_feature.detach()).to(device)
        del volume_feature

    def forward(self, dataset):
        self.MVSNet.train()
        num_views = 10
        image_preds_all = []

        for idx in range(num_views):
            batch = dataset[idx]
            rays, image_gt = self.decode_batch(batch)
            rays = rays.cuda()
            image_gt = image_gt.cpu()  # (H, W, 3)
            # mask = batch['mask'][0]

            N_rays_all = rays.shape[0]

            ##################  rendering #####################
            keys = ['val_psnr_all']
            log = init_log({}, keys)
            with torch.no_grad():
                image_preds, depth_preds = [],[]
                for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                        N_samples=args.N_samples, lindisp=args.use_disp)

                    # Converting world coordinate to ndc coordinate
                    H, W = image_gt.shape[:2]
                    inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                    w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0].clone()
                    intrinsic_ref[:2] *= args.imgScale_test/args.imgScale_train
                    xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                                near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad*args.imgScale_test, lindisp=args.use_disp)

                    # important sampleing
                    if self.density_volume is not None and args.N_importance > 0:
                        xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                        self.density_volume, z_vals,xyz_NDC,N_importance=args.N_importance)
                        xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                        near=self.near_far_source[0], far=self.near_far_source[1],pad=args.pad, lindisp=args.use_disp)


                    # rendering
                    rgb, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled,
                                                                        xyz_NDC, z_vals, rays_o, rays_d,
                                                                        self.volume, self.imgs,
                                                                        **self.render_kwargs_train)

                    image_preds.append(rgb.cpu());depth_preds.append(depth_pred.cpu())

                image_preds, depth_r = torch.clamp(torch.cat(image_preds).reshape(H, W, 3),0,1), torch.cat(depth_preds).reshape(H, W)
                image_preds_all += [image_preds]

                imageio.imwrite('visuals/{0}_{1}_pred_rgb.png'.format(dataset.scan, str(idx)), (image_preds.numpy()*255.).astype(np.uint8))
                imageio.imwrite('visuals/{0}_{1}_gt_rgb.png'.format(dataset.scan, str(idx)),(image_gt.numpy()*255.).astype(np.uint8))
        
        # demo
        frames = torch.stack(image_preds_all, 0) 
        imageio.mimwrite('visuals/video_{0}_{1}.mp4'.format(dataset.scan, str(idx)), (frames.cpu().numpy() * 255).astype(np.uint8), fps=30, quality=8)
                

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = MVSSystem(args)
    checkpoint_callback = ModelCheckpoint(os.path.join(f'runs_fine_tuning/{args.expname}/ckpts/','{epoch:02d}'),
                                            monitor='val/PSNR',
                                            mode='max',
                                            save_top_k=0)

    logger = loggers.TestTubeLogger(
        save_dir="runs_fine_tuning",
        name=args.expname,
        debug=False,
        create_git_tag=False
    )

    z_near = 2.
    z_far = 6.

    testsavedir = os.path.join('visuals')
    os.makedirs(testsavedir, exist_ok=True)

    if args.validate_nvs:
        print("Generating NVS")
        with torch.no_grad():
            system(system.val_dataset)

    if args.validate_mesh:
        scene = os.path.basename(args.datadir)
        print("Generating Mesh")
        with torch.no_grad():
            validate_mesh(scene=scene, savedir=testsavedir, **system.render_kwargs_train) # , scale_mats_np=scale_mats_np