#!/usr/bin/env python3
"""Calculates the Kernel Inception Distance (KID) to evalulate GANs
"""
import os
import pathlib
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from sklearn.metrics.pairwise import polynomial_kernel
from scipy import linalg
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
import ipdb
try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x
import cv2
from models.inception import InceptionV3
from models.lenet import LeNet5

def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False,reso=128):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    is_numpy = True if type(files[0]) == np.ndarray else False

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches), end='', flush=True)
        start = i * batch_size
        end = start + batch_size
        if is_numpy:
            images = np.copy(files[start:end]) + 1
            images /= 2.
        else:
            images=[]
            #ipdb.set_trace()
            for f in files[start:end]:
                try:
                    img=cv2.imread(str(f))
                    img=cv2.resize(img,(reso,reso),interpolation=cv2.INTER_CUBIC)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                except:
                    img=cv2.imread(str(files[0]))
                    img=cv2.resize(img,(reso,reso),interpolation=cv2.INTER_CUBIC)
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    print(str(f))
                images.append(img)
            #ipdb.set_trace()


            #images = [np.array(Image.open(str(f)).convert('RGB')) for f in files[start:end]]
            images = np.stack(images).astype(np.float32) / 255.
            # Reshape to (n_images, 3, height, width)
            images = images.transpose((0, 3, 1, 2))

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print('done', np.min(images))

    return pred_arr


def extract_lenet_features(imgs, net):
    net.eval()
    feats = []
    imgs = imgs.reshape([-1, 100] + list(imgs.shape[1:]))
    if imgs[0].min() < -0.001:
      imgs = (imgs + 1)/2.0
    print(imgs.shape, imgs.min(), imgs.max())
    imgs = torch.from_numpy(imgs)
    for i, images in enumerate(imgs):
        feats.append(net.extract_features(images).detach().cpu().numpy())
    feats = np.vstack(feats)
    return feats


def _compute_activations(path, model, batch_size, dims, cuda, model_type,reso,dataset):
    
    if not type(path) == np.ndarray:
        import glob
        import pathlib

        #ipdb.set_trace()

        files=[]

        path = pathlib.Path(path)
        for classname in os.listdir(path):
            classpath=os.path.join(path,classname)
            if os.path.isdir(classpath):
                for instance in os.listdir(classpath):
                    img=os.path.join(classpath,instance,'render','images')
                    if os.path.isdir(img):
                        files = files+sorted([os.path.join(img, idd) for idd in os.listdir(img) if idd.endswith('.png')])
    #files=files[:2]
    basepath=os.path.join(os.getcwd(),'results','kid'+str(reso))
    os.makedirs(os.path.join(basepath), exist_ok=True)
    #ipdb.set_trace()
    if model_type == 'inception':
        if os.path.exists(os.path.join(basepath,path.name+str(reso)+'kid.npy')):
            act=np.load(os.path.join(basepath,path.name+str(reso)+'kid.npy'))
            print('load_dataset',dataset)
        else:
            act = get_activations(files, model, batch_size, dims, cuda,reso=reso)
            np.save(os.path.join(basepath,path.name+str(reso)+'kid'),act)
    elif model_type == 'lenet':
        act = extract_lenet_features(files, model)
    #ipdb.set_trace()
    return act


def _compute_activations_new(path, model, batch_size, dims, cuda, model_type,reso,dataset):
    sample_name=path.split('/')[-1]
    if not type(path) == np.ndarray:
        import glob
        import pathlib


        files=[]

        path = pathlib.Path(path)
        for classname in os.listdir(path):
            classpath=os.path.join(path,classname)

            
            for instance in os.listdir(classpath):
                if os.path.isdir(os.path.join(classpath,instance)):
                    img=os.path.join(classpath,instance)
                    files = files+sorted([os.path.join(img, idd) for idd in os.listdir(img) if idd.endswith('.png')])
    
    
    basepath=os.path.join(os.getcwd(),'results','kid'+str(reso))
    os.makedirs(os.path.join(basepath), exist_ok=True)

    if model_type == 'inception':
        if os.path.exists(os.path.join(basepath,str(reso)+'kid.npy')):
            act=np.load(os.path.join(basepath,str(reso)+'kid.npy'))
            print('load_sample')
        else:
            act = get_activations(files, model, batch_size, dims, cuda,reso=reso)
            np.save(os.path.join(basepath,str(reso)+'kid'),act)
    elif model_type == 'lenet':
        act = extract_lenet_features(files, model)

    return act

def calculate_kid_given_paths(paths, batch_size, cuda, dims, model_type='inception',reso=128,dataset='oo3d'):
    """Calculates the KID of two paths"""
    pths = []
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
        if os.path.isdir(p):
            pths.append(p)
        # elif p.endswith('.npy'):
        #     np_imgs = np.load(p)
        #     if np_imgs.shape[0] > 50000: np_imgs = np_imgs[np.random.permutation(np.arange(np_imgs.shape[0]))][:50000]
        #     pths.append(np_imgs)

    if model_type == 'inception':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
    elif model_type == 'lenet':
        model = LeNet5()
        model.load_state_dict(torch.load('./models/lenet.pth'))
    if cuda:
       model.cuda()

   
    actj = _compute_activations_new(pths[0], model, batch_size, dims, cuda, model_type,reso,dataset)
        


def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)




if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fake', type=str, nargs='+', required=True,
                        help=('Path to the generated images'))
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size to use')
    parser.add_argument('--reso', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('-c', '--gpu', default='', type=str,
                        help='GPU to use (leave blank for CPU only)')
    parser.add_argument('--model', default='inception', type=str,
                        help='inception or lenet')
    parser.add_argument('--dataset', default='oo3d', type=str,
                        help='inception or lenet')
    args = parser.parse_args()
    print(args)
    #ipdb.set_trace()
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    paths = args.fake
    

    calculate_kid_given_paths(paths, args.batch_size,True, args.dims, model_type=args.model,reso=args.reso,dataset=args.dataset)
    
