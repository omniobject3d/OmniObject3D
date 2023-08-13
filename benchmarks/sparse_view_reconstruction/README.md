# Sparse-view reconstruction Benchmark

This is a short documentation of 3D reconstruction and novel view synthesis (NVS) given sparse-view observations on OmniObject3D.

## :floppy_disk: Data
- Download our public dataset by
```
odl login
odl get OpenXD-OmniObject3D-New/raw/blender_renders
```
- In our paper, we select the `first 3` objects from the each of the following ten categories for evaluation: 
```
['toy_train', 'bread', 'cake', 'toy_boat', 'hot_dog', 'wallet', 'pitaya', 'squash', 'handbag', 'apple']
```
- To attend our [challenge @ ICCV 2023](https://omniobject3d.github.io/challenge.html), the hidden test set can be obtained [here](https://drive.google.com/file/d/1GKEa-r1__tnVKAZSF5I5uWcLh1cAllqh/view?usp=drive_link).


## :hammer: Training
OmniObject3D implements multiple generalizable NVS methods using official codebases of [PixelNeRF](https://github.com/sxyu/pixel-nerf), [MVSNeRF](https://github.com/apchenstu/mvsnerf), and [IBRNet](https://github.com/googleinterns/IBRNet). We use default hyperparameters for experiments, except for the parameter changes described in [Supp. C.2.2](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wu_OmniObject3D_Large-Vocabulary_3D_CVPR_2023_supplemental.pdf). For more technical details of these methods, please check the corresponding papers. 
For a quick start, we provide a brief introduction, dataloaders (to be replaced in the official codebase), and examples on how to use them in the following.

### PixelNeRF
PixelNeRF uses an CNN encoder to compute feature volume on the input image and conditions a volume rendering framework where these features are aggregated over multiple views to output radiance and density.
```
# dataloader
├── _pixelnerf 
    ├── conf
        ├── exp
            ├── oo3d.conf
    ├── src
        ├── data
            ├── __init__.py
            ├── OO3DDataset.py
    ├── eval
        ├── tools
            ├── mesh_eval.py
            ├── process_testsets.py
        ├── gen_results.py
        ├── gen_testset_results.py

# prepare the code
git clone git@github.com:sxyu/pixel-nerf.git
cp -r -f _pixelnerf/* pixel-nerf/
```
```
# pretraining
python train/train.py \
    -n oo3d \
    -c conf/exp/oo3d.conf \
    -D path/to/omniobject3d \
    -V 3 \ # source views
    --gpu_id='0 1 2 3' 
```

### MVSNeRF
MVSNeRF constructs a cost volume by warping 2D image features onto a plane sweep and then applys 3D CNN to reconstruct a neural encoding volume that can be further conditioned on an MLP to regress density and color.
```
# dataloader
├── _mvsnerf
    ├── opt.py
    ├── model.py
    ├── data
        ├── __init__.py
        ├── oo3d.py
        ├── oo3d_ft.py
        ├── oo3d_testsub.py
    ├── poses.json
    ├── ref_src_pairs.json
    ├── generate_results.py
    ├── generate_testset_results.py
    ├── tools
        ├── mesh_eval.py
        ├── process_testsets.py

# prepare the code
git clone git@github.com:apchenstu/mvsnerf.git
cp -r -f _mvsnerf/* mvsnerf/
```
Download 'poses.json' and 'ref_src_pairs.json' at [here](https://drive.google.com/drive/folders/1vxJuPFYdqhiayBzO0FU8ZAUlThB93esy?usp=drive_link) and then put them at 'mvsnerf/'.
```
# pretraining
python train_mvs_nerf_pl.py \
    --expname oo3d \
    --use_viewdirs \
    --dataset_name oo3d \
    --num_gpus 1 \
    --batch_size 1024 \
    --datadir path/to/omniobject3d

# finetuning
python train_mvs_nerf_finetuning_pl.py  \
    --dataset_name oo3d_ft \
    --datadir path/to/omniobject3d/category/category_00S \ # Sth object in category
    --expname oo3d_ft_category \
    --with_rgb_loss   \
    --batch_size 1024  \
    --num_epochs 1 \
    --ckpt path/to/pretrained-ckpt \
    --N_vis 1
```


### IBRNet
IBRNet represents each image as a individual scene and applys transformer to aggregate features interpolated from a set of nearby sources to reason underlying color and density.
```
# dataloader
├── _ibrnet
    ├── config.py
    ├── train.py
    ├── poses.json
    ├── ref_src_pairs.json
    ├── configs
        ├── oo3d.txt
        ├── oo3d_ft.txt
    ├── ibrnet
        ├── data_loaders
            ├── __init__.py
            ├── oo3d.py
            ├── oo3d_ft.py
            ├── oo3d_testsub.py
            ├── ray_utils.py
            ├── create_training_dataset.py
    ├── eval
        ├── eval.py
        ├── eval_testset.py

# prepare the code
git clone git@github.com:googleinterns/IBRNet.git
cp -r -f _ibrnet//* IBRNet/
```
Download 'poses.json' and 'ref_src_pairs.json' at [here](https://drive.google.com/drive/folders/1vxJuPFYdqhiayBzO0FU8ZAUlThB93esy?usp=drive_link) and then put them at 'IBRNet/'.
```
# pretraining
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/oo3d.txt

# finetuning
python -m torch.distributed.launch --nproc_per_node=2 train.py --config configs/oo3d_ft.txt
```

## :microscope: Evaluation
We provide scripts to generate novel appearance and extract the underlying mesh from implicit density field. Note that all methods use 3 source views by default.
If you are interested in attending our [challenge](https://omniobject3d.github.io/challenge.html) in the sparse-view reconstruction track, we also provide scripts to generate results on the hidden test set in the standard format for submission.

### PixelNeRF
```
# novel view rendering and extracting mesh
python eval/gen_results.py  \
    -n oo3d \
    --gpu_id='0' \
    --split test \
    -c conf/exp/oo3d.conf \
    -D  path/to/omniobject3d/category \
    -S 0 \ # Sth object in category
    --scale 1 \
    --validate_nvs \
    --validate_mesh
```
```
# generate results on the hidden test set for submission
python eval/gen_testset_results.py  \
    -n oo3d \
    --gpu_id='0' \
    --split testsub \
    -c conf/exp/oo3d.conf \
    -D  path/to/the/hidden/testset # run through the whole test set
    --output_dir path/to/output
```

The standard output format for the challenge submission would be:
```
-- output # do NOT rename this
---- obj_000
------ images
------ pcd.ply
---- obj_001
...
---- obj_079
```
Compress the folder into a zip file and upload it! An example of a successful submission can be found [here](https://drive.google.com/file/d/1HtxMvdqoAKTQsW5bTja9n3h5eJT24kCk/view?usp=drive_link).

### MVSNeRF
```
# novel view rendering and extracting mesh
python generate_results.py  \
    --dataset_name oo3d_ft \
    --datadir path/to/omniobject3d/category/category_00S \ # Sth object in category
    --expname oo3d \
    --ckpt path/to/your_ckpt \
    --validate_nvs \
    --validate_mesh
```

```
# generate results on the hidden test set for submission
python generate_testset_results.py  \
    --dataset_name oo3d_testsub \
    --datadir path/to/the/hidden/testset/obj_00S \ # Sth object in the test set
    --expname oo3d \
    --ckpt path/to/your_ckpt \
    --output_dir path/to/output \
    --use_testsub \
    --validate_nvs \
    --validate_mesh
```

### IBRNet
```
# novel view rendering
cd eval/
python eval.py  \
    --config configs/oo3d.txt \
    --datadir path/to/omniobject3d/category/category_00S \ # Sth object in category
    --train_dataset oo3d_ft \
    --ckpt_path path/to/your_ckpt

# generate results on the hidden test set for submission
cd eval/
python eval_testset.py  \
    --config ../configs/oo3d.txt \
    --datadir path/to/the/hidden/testset/obj_00S \ # Sth object in the test set
    --train_dataset oo3d_testsub \
    --use_testsub \
    --output_dir path/to/output \
    --ckpt_path path/to/your_ckpt
```
