# OmniObject3D Toolbox


<div align="center">

<h1>OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation
</h1>

<div>
    <a href='https://wutong16.github.io/' target='_blank'>Tong Wu</a>&emsp;
    Jiarui Zhang&emsp;
    <a href='https://fuxiao0719.github.io/' target='_blank'>Xiao Fu</a>&emsp;
    Yuxin Wang&emsp;
    <a href='https://jiawei-ren.github.io/' target='_blank'>Jiawei Ren</a>&emsp;
    <a href='https://scholar.google.com/citations?user=lSDISOcAAAAJ&hl=zh-CN' target='_blank'>Liang Pan</a>&emsp;
    <a href='https://wywu.github.io/' target='_blank'>Wayne Wu</a>&emsp;
    <a href='https://scholar.google.com.hk/citations?user=jZH2IPYAAAAJ&hl=en' target='_blank'>Lei Yang</a>&emsp;
    <a href='https://myownskyw7.github.io/' target='_blank'>Jiaqi Wang</a>&emsp;
    <a href='https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&hl=zh-CN&user=AerkT0YAAAAJ&sortby=pubdate' target='_blank'>Chen Qian</a>&emsp;
    <a href='https://scholar.google.com/citations?user=GMzzRRUAAAAJ&hl=zh-CN' target='_blank'>Dahua Lin</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>&emsp;
</div>

<strong>Accepted to <a href='https://cvpr2023.thecvf.com/' target='_blank'>CVPR 2023</a> as Award Candidate </strong> :partying_face:

<strong><a href='https://omniobject3d.github.io/' target='_blank'>Project</a>&emsp;</strong>
<strong><a href='https://arxiv.org/abs/2208.12697' target='_blank'>Paper</a>&emsp;</strong>
<strong><a href='https://opendatalab.com/OpenXD-OmniObject3D-New/download' target='_blank'>Data</a></strong>
</div>

![colored_mesh (1)](assets/teaser.png)

## Updates
- [06/2023] We release the training set of OmniObject3D (90\% of the whole data), keeping a hidden testing set for an upcoming challenge. Stay tuned!
## Usage
### Download the dataset
- Sign up [here](https://opendatalab.com/OpenXD-OmniObject3D-New/download).
- Install OpenDataLab's CLI tools through `pip install opendatalab`.
- View and download the dataset from the command line:
```
odl login                                 # Login
odl ls     OpenXD-OmniObject3D-New        # View a list of dataset files
odl get    OpenXD-OmniObject3D-New        # Download the whole dataset (the compressed files require approximately 1.2TB of storage)
```
You can check out the full folder structure on the website above and download a certain portion of the data by specifying the path. For example:
```
odl get OpenXD-OmniObject3D-New/raw/point_clouds/ply_files
```

### Batch untar
To batch-untar a specific folder of compressed files based on your requirements, use the command `bash batch_untar.sh <folder_name>`. 
If the untar operation is completed successfully, remove all compressed files through `rm -rf <folder_name>/*.tar.gz`.

### Dataset format

```
OmniObject3D_Data_Root
    ├── raw_scans               
    │   ├── <category_name>
    │   │   ├── <object_id>
    │   │   │   ├── Scan
    │   │   │   │   ├── Scan.obj
    │   │   │   │   ├── Scan.mtl
    │   │   │   │   ├── Scan.jpg
    
    ├── blender_renders         
    │   ├── <category_name>
    │   │   ├── <object_id>
    │   │   │   ├── render
    │   │   │   │   ├── images
    │   │   │   │   ├── depths
    │   │   │   │   ├── normals
    │   │   │   │   ├── transforms.json    
    
    ├── videos_processed       
    │   ├── <category_name>
    │   │   ├── <object_id>
    │   │   │   ├── standard
    │   │   │   │   ├── images
    │   │   │   │   ├── matting
    │   │   │   │   ├── poses_bounds.npy           # raw results from colmap
    │   │   │   │   ├── poses_bounds_rescaled.npy  # rescaled to world-scale
    │   │   │   │   ├── sparse

    ├── point_clouds    
    │   ├── hdf5_files
    │   │   ├── 1024
    │   │   ├── 4096
    │   │   ├── 16384
    │   ├── ply_files
    │   │   ├── 1024
    │   │   ├── 4096
    │   │   ├── 16384
```

## License
The OmniObject3D dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Reference
If you find our dataset useful in your research, please use the following citation:
```
@inproceedings{wu2023omniobject3d,
    author = {Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, 
    Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, Ziwei Liu},
    title = {OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation},
    journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
    }
```
