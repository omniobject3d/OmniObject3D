import os
import numpy as np
import argparse
import h5py
import trimesh
from plyfile import PlyData,PlyElement
try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.ops import sample_farthest_points
except:
    print('pytorch3D not imported')

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_path = os.path.join(os.path.dirname(pyexample_path), 'test_data')


def write_ply(save_path,points,text=True):
    """
    save_path : path to save: '/yy/XX.ply'
    pt: point_cloud: size (N,3)
    """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(save_path)

def sample_and_visualize(path, num_points=8192, save_dir=''):
    try:
        mesh = trimesh.load_mesh(path)
        pcd = trimesh.sample.sample_surface(mesh, num_points)[0]
    except:
        print('{} failed!! '.format(path))
        pcd = None
    if pcd is not None:
        write_ply(save_dir, pcd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_dir', type=str, default='none')
    parser.add_argument('--output_dir', type=str, default='point_clouds')
    parser.add_argument('--output_dir_h5', type=str, default='point_clouds')
    parser.add_argument('--frames_dir', type=str, default='none')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--obj2ply', action='store_true')
    parser.add_argument('--ply2h5', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    categories = []
    if args.obj2ply:
        cats = sorted(os.listdir(args.model_dir))
        for cat in cats:
            if not args.all and cat not in categories or not os.path.isdir(os.path.join(args.model_dir, cat)):
                continue

            print('= = = = = {} starts! = = = = ='.format(cat))
            objs = sorted(os.listdir(os.path.join(args.model_dir, cat)))
            for obj in objs:
                path = os.path.join(args.model_dir, cat, obj, 'Scan')
                for file in os.listdir(path):
                    if not 'obj' in file:
                        continue
                    filename = os.path.join(path, file)
                    output_folder = os.path.join(args.output_dir, cat, obj)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder, exist_ok=True)
                    elif len(os.listdir(output_folder)) > 0: # already extracted pcd
                        continue
                    save_dir = os.path.join(output_folder, 'pcd_{}.ply'.format(args.num_points))
                    sample_and_visualize(filename, num_points=args.num_points, save_dir=save_dir)

    if args.ply2h5:
        cats = sorted(os.listdir(args.output_dir))
        print('categories:', cats)
        for i, cat in enumerate(cats):
            objs = os.listdir(os.path.join(args.output_dir, cat))
            point_clouds = []
            for obj in objs:
                pcd_file = os.path.join(args.output_dir, cat, obj, 'pcd_{}.ply'.format(args.num_points))
                try:
                    pcd = np.array(trimesh.load(pcd_file).vertices)
                    point_clouds.append(pcd)
                except:
                    print('Failed to load {}, skip it'.format(pcd_file))
            point_clouds = np.stack(point_clouds)
            file = os.path.join(args.output_dir_h5, '{}_{}.hdf5'.format(cat, args.num_points))
            with h5py.File(file, 'w') as f:
                f.create_dataset("data", data=point_clouds)
            print('{} done, saved to {}'.format(cat, file))
