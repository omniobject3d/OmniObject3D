# 3D Object Generation Benchmark

This is a short documentation of OmniObject3D's benchmark, 3D object generation, organized as follows:

- [3D Object Generation Benchmark](#3d-object-generation-benchmark)
  - [Data](##Data)
  - [Training](##Training)
    - [GET3D](###GET3D)
  - [Evaluation](##Evaluation)

## Data
We provide the raw scans, 100-view rendered images (under the [nerf_synthetic dataset](https://github.com/bmild/nerf) format), and 24-view rendered images following the format of [GET3D](https://github.com/nv-tlabs/GET3D). The data could be downloaded following the instructions [here](https://github.com/omniobject3d/OmniObject3D#download-the-dataset).

## Training
OmniObject3D implements 3D object generation using the official codebase of [GET3D](https://github.com/nv-tlabs/GET3D). We use default hyperparameters for the experiments. 
For a quick start, we provide brief introduction, dataloaders (to be replaced in the official codebase), and examples on how to use them in the following.

### GET3D
GET3D is a generative model that directly generates explicit textured 3D meshes with complex topology, rich geometric details, and high fidelity textures.
```
# data loader
├── _get3d 
    ├── train_3d.py
    ├── training
        ├── dataset.py
        ├── discriminator_architecture.py
        ├── inference_3d.py
        ├── inference_utils.py
        ├── loss.py
        ├── networks_get3d.py
        ├── sample_camera_distribution.py
    
# clone from the official repo
git clone git@github.com:nv-tlabs/GET3D.git

# copy and overwrite some files in the original repo
cp -r -f _get3d/* GET3D/

# setup the environment following their instructions.

# training 
python train_3d.py --outdir=PATH_TO_LOG \
    --data=PATH_TO_RENDER_IMG \
    --camera_path PATH_TO_RENDER_CAMERA \
    --gpus=8 --batch=32 --gamma=40 \
    --data_camera_mode oo3d \  
    --dmtet_scale 1.0  \
    --use_shapenet_split 1  \
    --one_3d_generator 1  \
    --fp32 0
```

## Evaluation
If you are interested in attending our [challenge]() in the generation track, we provide instructions on preparing the outputs for submission in the standard format.


First, randomly generate 5,000 objects and render each of them from 10 randomly sampled view points for 2D evaluation:
```
---- results_img
------ 0
-------- 0.png
-------- 1.png
-------- ...
-------- 9.png
------ 1
------ 2
...
------ 4999
```
Then, randomly pick 1,000 objects for 3D evaluation:
```
---- results_mesh
------ 0.obj
------ 1.obj
------ ...
------ 999.obj
```

As for GET3D, the preparation could be done by running: 
```
python train_3d.py \
--data PATH_TO_RENDER_IMG \
--camera_path PATH_TO_RENDER_CAMERA \ 
--inference_mode competition \
--outdir ./save_inference_results/example_output \  
--gpus 1 --batch 4 --gamma 40 \
--data_camera_mode oo3d \
--dmtet_scale 1.0 \
--one_3d_generator 1 \
--fp32 0 \
--inference_vis 1 \
--resume_pretrain PATH_TO_CHECKPOINT \
--inference_save_interpolation 1 \
--cond 0 \
--n_forimg 5000 \
--n_forgeo 1000 \

```

Finally, run the following scripts to generate the output files:
```
python fid_score.py path/to/your/results_img --reso 128 --save_path ./my_results

python compute_cd_mmd.py path/to/your/results_mesh --n_points 2048 --save_path ./my_results
```

The standard output format would be:
```
-- my_results
---- covmmd
---- fid128
```
