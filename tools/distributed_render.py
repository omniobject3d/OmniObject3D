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

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
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
            f" --obj_path {item}"
        )
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    # python distributed_render.py --num_gpus 2 --workers_per_gpu 2
    args = tyro.cli(Args)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()

    omni3d_root = "/home/shenqiuhong/dataset/omniobject"
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
                queue.put(instance_path)


    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
    
    print("All objects rendered !")
