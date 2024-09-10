import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import tyro
import os 
from os import path as osp

import pdb


@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    # input_models_path: str
    # """Path to a json file containing a list of 3D object files"""
    data_root: str
    """dataset rootpath"""

    output_dir: str
    """output rootpath"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""

    limit_num: int = -1
    """num of objects limit. -1 mean no limit"""


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    out_dir: str,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        # Perform some operation on the item
        print(item, gpu)
        command = (
            f"export DISPLAY=:0.{gpu} &&"
            f" blender -b -P ./blender_script.py --"
            f" --obj_path {item} --output {out_dir}"
        )
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    # python distributed_render.py --num_gpus 8 --workers_per_gpu 2 --data_root /local_home/shenqiuhong/omni3d/raw_scan --limit_num 500 --output_dir /local_home/shenqiuhong/omni_render/
    args = tyro.cli(Args)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    out_dir = args.output_dir
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, out_dir)
            )
            process.daemon = True
            process.start()

    omni3d_root = args.data_root

    instance_list = []

    for category in os.listdir(omni3d_root):
        category_path = osp.join(omni3d_root, category)
        for instance in os.listdir(category_path):
            if '.txt' in instance:
                instance_path = osp.join(category_path, instance)
                os.remove(instance_path)
            else:
                instance_path = osp.join(category, instance)
                instance_list.append(instance_path)

    limit_num = len(instance_list) if args.limit_num < 0 else args.limit_num
    instance_list = sorted(instance_list)
    instance_list = instance_list[:limit_num]
    for instance_path in instance_list:
        obj_path = osp.join(omni3d_root, instance_path)
        queue.put(obj_path)


    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
    
    print("All objects rendered !")
