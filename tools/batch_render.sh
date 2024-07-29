#! /bin/bash
python distributed_render.py --num_gpus 8 --workers_per_gpu 2 \
    --data_root /local_home/shenqiuhong/omni3d/raw_scan \
    --output_dir /local_home/shenqiuhong/omni_render/ \
    --limit_num 500 