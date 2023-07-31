# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from . import dataset_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
from typing import Optional
from operator import itemgetter
import torch


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def create_training_dataset(args, mode):
    # parse args.train_dataset, "+" indicates that multiple datasets are used, for example "ibrnet_collect+llff+spaces"
    # otherwise only one dataset is used
    # args.dataset_weights should be a list representing the resampling rate for each dataset, and should sum up to 1

    mode = mode
    if mode == 'train':
        print('training dataset: {}'.format(args.train_dataset))
    else:
        print('validation dataset: {}'.format(args.train_dataset))
    if '+' not in args.train_dataset:
        train_dataset = dataset_dict[args.train_dataset](args, mode)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        train_dataset_names = args.train_dataset.split('+')
        weights = args.dataset_weights
        assert len(train_dataset_names) == len(weights)
        assert np.abs(np.sum(weights) - 1.) < 1e-6
        print('weights:{}'.format(weights))
        train_datasets = []
        train_weights_samples = []
        for training_dataset_name, weight in zip(train_dataset_names, weights):
            train_dataset = dataset_dict[training_dataset_name](args, mode,
                                                                scenes=args.train_scenes,
                                                                )
            train_datasets.append(train_dataset)
            num_samples = len(train_dataset)
            weight_each_sample = weight / num_samples
            train_weights_samples.extend([weight_each_sample]*num_samples)

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_weights = torch.from_numpy(np.array(train_weights_samples))
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_sampler = DistributedSamplerWrapper(sampler) if args.distributed else sampler

    return train_dataset, train_sampler

