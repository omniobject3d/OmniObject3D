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
from scipy.interpolate import CubicSpline
from scipy.spatial import distance_matrix
import tqdm
import cv2
from torch.autograd import Variable
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import lpips
import matplotlib.pyplot as plt
loss_fn_vgg = lpips.LPIPS(net='vgg')
import pytorch_msssim
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def extra_args(parser):
    parser.add_argument(
        "--subset", "-S", type=int, default=0, help="Subset in data to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
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
        "--num_views",
        type=int,
        default=40,
        help="Number of video frames (rotated views)",
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
    parser.add_argument("--fps", type=int, default=30, help="FPS of video")
    parser.add_argument("--validate_mesh", action='store_true')
    parser.add_argument("--validate_nvs", action='store_true')
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])

dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)

data = dset[args.subset]
data_path = data["path"]
print("Data instance loaded:", data_path)

images = data["images"]  # (NV, 3, H, W)
poses = data["poses"]  # (NV, 4, 4)
poses_origin = data['poses_origin']

def read_depth(filename):
    depth_h = cv2.imread(filename ,cv2.IMREAD_UNCHANGED)[...,-1]
    depth_h[depth_h==65504.] = 0.
    downSample = 0.5
    depth = cv2.resize(depth_h, None, fx=downSample, fy=downSample, interpolation=cv2.INTER_NEAREST)  
    mask = depth > 0
    return depth

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
        centroids[:, i] = farthest                                      
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)        
        dist = torch.sum((xyz - centroid) ** 2, -1)                     
        mask = dist < distance
        distance[mask] = dist[mask]                                    
        farthest = torch.max(distance, -1)[1]
    
    return centroids

# ensure the same sampled frames with mvsnerf and ibrnet
xyzs = data["poses_origin"][:,:3,-1]
distances = distance_matrix(xyzs, xyzs, p=2)
rank = np.argsort(distances[50])
num_select = 30
xyzs = xyzs[rank[:num_select]]
rank = rank[:num_select]
num_src = args.num_src
centroids = farthest_point_sample(torch.from_numpy(xyzs[None]), num_src)
source = [ rank[centroids[0][i].item()] for i in range(num_src)]
source = torch.Tensor(source).to(torch.long)
xyzs = data["poses_origin"][:,:3,-1]
distances = distance_matrix(xyzs, xyzs, p=2)
rank = np.argsort(distances[50])
centroids = farthest_point_sample(torch.from_numpy(xyzs[None]), 100)
ref_pair_idx = [ centroids[0][i].item() for i in range(100)]
for i in range(num_src):
    ref_pair_idx.remove(source[i])
render_poses_idx = ref_pair_idx[16-num_src : 26-num_src]
data_bbox = data['bbox'][render_poses_idx]
render_poses = poses[render_poses_idx]
print("Sampled source: ", source)

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

if args.scale != 1.0:
    Ht = int(H * args.scale)
    Wt = int(W * args.scale)
    if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
        warnings.warn(
            "Inexact scaling, please check {} times ({}, {}) is integral".format(
                args.scale, H, W
            )
        )
    H, W = Ht, Wt

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

def validate_mesh(resolution=512, scene=0, savedir='', threshold=2., cat=None, scale_mats_np=None, network_query_fn=None):
    import trimesh
    os.makedirs(savedir, exist_ok=True)
    vertices, triangles = extract_geometry(resolution=resolution, threshold=threshold, network_query_fn=network_query_fn)
    if scale_mats_np is not None:
        vertices = vertices * scale_mats_np[0, 0] + scale_mats_np[:3, 3][None]
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh_path = os.path.join(savedir, cat+"_{:03d}".format(scene+1)+'.ply')
    mesh.export(mesh_path)
    print("mesh saved at " + mesh_path)

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
random_source = NS == 1 and source[0] == -1
assert not (source >= NV).any()

if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
    renderer.n_fine = 128

with torch.no_grad():
    print("Encoding source view(s)")
    if random_source:
        src_view = torch.randint(0, NV, (1,))
    else:
        src_view = source

    net.encode(
        images[src_view].unsqueeze(0),
        poses[src_view].unsqueeze(0).to(device=device),
        focal,
        c=c,
    )

    os.makedirs('visuals', exist_ok=True)

    # render demo views
    if args.validate_nvs:
        print("Rendering",len(render_poses_idx) * H * W, "rays")
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

        num_views = len(render_poses_idx)
        for i in range(num_views):
            idx = render_poses_idx[i]
            image_gt = images[idx].permute(1,2,0) *0.5 + 0.5
            image_pred = frames[i].cpu()
            
            imageio.imwrite('visuals/{0}_{1}_{2}_pred_rgb.png'.format(str(args.subset), str(i), os.path.basename(dset.base_path)), (image_pred*255.).numpy().astype(np.uint8))
            imageio.imwrite('visuals/{0}_{1}_{2}_gt_rgb.png'.format(str(args.subset), str(i), os.path.basename(dset.base_path)), (image_gt*255.).numpy().astype(np.uint8))

        print("Writing video")
        vid_name = "{:02}".format(args.subset)
        vid_path = os.path.join('visuals', "video_" + vid_name + ".mp4")
        imageio.mimwrite(vid_path, (frames.cpu().numpy() * 255).astype(np.uint8), fps=30, quality=8)
    
    if args.validate_mesh:
        print("Generating Mesh")
        cat = os.path.basename(dset.base_path)
        with torch.no_grad():
            testsavedir = os.path.join('visuals')
            if os.path.exists(testsavedir) == False:
                os.mkdir(testsavedir)
            validate_mesh(scene=args.subset, savedir=testsavedir, cat=cat, network_query_fn=render_par) # , scale_mats_np=scale_mats_np
