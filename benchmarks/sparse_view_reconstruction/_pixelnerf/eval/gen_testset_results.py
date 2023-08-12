import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn.functional as F
import numpy as np
import imageio
import util
import warnings
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
import tqdm
import cv2
from termcolor import colored
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import lpips
import matplotlib.pyplot as plt
loss_fn_vgg = lpips.LPIPS(net='vgg')
from tools.mesh_eval import read_and_downsample_pcd_from_mesh

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def validate_mesh(resolution=256, savedir='', threshold=2., cat=None, scale_mats_np=None, network_query_fn=None):
    import trimesh
    os.makedirs(savedir, exist_ok=True)
    vertices, triangles = extract_geometry(resolution=resolution, threshold=threshold, network_query_fn=network_query_fn)
    if scale_mats_np is not None:
        vertices = vertices * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(os.path.join(savedir, 'mesh.ply'))
    data_down = read_and_downsample_pcd_from_mesh(os.path.join(savedir, 'mesh.ply'), thresh=0.01) # 0.01 for the sparse-view setting as default
    trimesh.PointCloud(data_down).export(os.path.join(savedir, 'pcd.ply'))
    os.remove(os.path.join(savedir, 'mesh.ply'))
    print("pcd saved at " + os.path.join(savedir, 'pcd.ply'), ', mesh removed.')

def extract_geometry(network_query_fn=None, resolution=128, threshold=0.001):
    bound_min = torch.tensor([-1., -1., -1.]).cuda() * 0.85
    bound_max = torch.tensor([1, 1, 1]).cuda() * 0.85
    def query_func(pts):
        viewdirs = torch.zeros_like(pts).cuda()
        raw = network_query_fn.net(pts[None].cuda(), coarse=True, viewdirs=viewdirs[None])
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

def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="testsub",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to do random",
    )
    parser.add_argument(
        "--num_src",
        type=int,
        default=3,
        help="Number of source frames",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=-10.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Distance of camera from origin, default is average of z_far, z_near of dataset (only for non-DTU)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='example_output_80_new_2',
        help="Where to save the results",
    )
    parser.add_argument("--validate_mesh", action='store_true')
    parser.add_argument("--validate_nvs", action='store_true')
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)

for i, data in enumerate(dset):
    data_path = data["path"]
    print("Data instance loaded:", data_path)

    images = data["images"]  # (NV, 3, H, W)
    poses = data["poses"]  # (NV, 4, 4)
    poses_origin = data['poses_origin']
    render_poses = data["render_poses"]
    test_img_files = data["test_img_files"]

    focal = data["focal"]
    if isinstance(focal, float):
        # Dataset implementations are not consistent about
        # returning float or scalar tensor in case of fx=fy
        focal = torch.tensor(focal, dtype=torch.float32)
    focal = focal[None]

    c = data.get("c")
    if c is not None:
        c = c.to(device=device).unsqueeze(0)

    NV, _, H, W = images.shape

    source = np.arange(NV)
    print("Sampled source: ", source)

    net = make_model(conf["model"]).to(device=device)
    net.load_weights(args)

    renderer = NeRFRenderer.from_conf(
        conf["renderer"], lindisp=dset.lindisp, eval_batch_size=args.ray_batch_size,
    ).to(device=device)

    render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

    # Get the distance from camera to origin
    z_near = dset.z_near
    z_far = dset.z_far

    print("Generating rays")
    render_rays = util.gen_rays(
        render_poses,
        W,
        H,
        focal * args.scale,
        z_near,
        z_far,
        c=c * args.scale if c is not None else None,
    ).to(device=device)
    # (NV, H, W, 8)

    focal = focal.to(device=device)

    # source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
    NS = len(source)

    if renderer.n_coarse < 64:
        # Ensure decent sampling resolution
        renderer.n_coarse = 64
        renderer.n_fine = 128

    with torch.no_grad():
        print("Encoding source view(s)")
        src_view = source

        net.encode(
            images[src_view].unsqueeze(0),
            poses[src_view].unsqueeze(0).to(device=device),
            focal,
            c=c,
        )

        if args.validate_nvs:
            # render test views
            print("Rendering",len(render_poses) * H * W, "rays")
            all_rgb_fine = []
            all_depth = []
            for rays in tqdm.tqdm(
                torch.split(render_rays.view(-1, 8), args.ray_batch_size, dim=0)
            ):
                rgb, _depth = render_par(rays[None])
                all_rgb_fine.append(rgb[0])
                all_depth.append(_depth[0])
            _depth = None
            rgb_fine = torch.cat(all_rgb_fine)
            all_depth = torch.cat(all_depth)
            frames = rgb_fine.view(-1, H, W, 3)
            depths = all_depth.view(-1, H, W)

            os.makedirs(os.path.join(args.output_dir, 'obj_{:03}'.format(i), 'images'), exist_ok=True)
            for j, frame in enumerate(frames):
                imageio.imwrite(os.path.join(args.output_dir, 'obj_{:03}'.format(i), 'images', test_img_files[j]), (frame.cpu().numpy() * 255).astype(np.uint8))
        
        if args.validate_mesh:
            print("Generating Mesh")
            with torch.no_grad():
                validate_mesh(savedir=os.path.join(args.output_dir, 'obj_{:03}'.format(i)), network_query_fn=render_par)
