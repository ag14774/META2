import math
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from utils.util import (get_global_rank, get_local_size, get_world_size,
                        is_distributed)


class OptionallyDistributedSubsetSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """
    def __init__(self, indices, shuffle=True):
        self.num_replicas = get_world_size()
        # if num_replicas is None:
        #     if not dist.is_available():
        #         raise RuntimeError(
        #             "Requires distributed package to be available")
        self.rank = get_global_rank()
        # if rank is None:
        #     if not dist.is_available():
        #         raise RuntimeError(
        #             "Requires distributed package to be available")
        self.shuffle = shuffle
        self.indices = indices
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = [
                self.indices[i]
                for i in torch.randperm(len(self.indices), generator=g)
            ]
        else:
            indices = [self.indices[i] for i in range(len(self.indices))]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class _BaseDataLoader(DataLoader):
    def __init__(self, sampler, **kwargs):
        self.sampler = sampler
        super(_BaseDataLoader, self).__init__(sampler=sampler, **kwargs)

    @abstractmethod
    def step(self, epoch):
        '''
        Can be used to store an internal state in the dataset
        Useful for iterative based training instead of epoch based
        '''
        self.sampler.set_epoch(epoch)


class BaseDataLoader(_BaseDataLoader, metaclass=ABCMeta):
    """
    Base class for all data loaders
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 validation_split,
                 num_workers,
                 drop_last=False,
                 collate_fn=default_collate,
                 pin_memory=True):

        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        if is_distributed():
            batch_size = int(batch_size / get_world_size())
            num_workers = int(num_workers / get_local_size())

        # If in distributed setting then we split the batch_size
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'drop_last': drop_last,
            'pin_memory': pin_memory
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler,
                                             **self.init_kwargs)

    def _split_sampler(self, split):
        idx_full = np.arange(self.n_samples)
        shuffle = self.shuffle

        if self.shuffle:
            random = np.random.RandomState(seed=0)
            random.shuffle(idx_full)

        if split == 0.0:
            self.shuffle = False
            train_sampler = OptionallyDistributedSubsetSampler(
                idx_full, shuffle)
            self.n_samples = len(train_sampler)
            return train_sampler, None

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, (
                "validation set size is configured"
                " to be larger than entire dataset.")
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = OptionallyDistributedSubsetSampler(train_idx, shuffle)
        valid_sampler = OptionallyDistributedSubsetSampler(valid_idx, shuffle)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_sampler)

        return train_sampler, valid_sampler

    @abstractmethod
    def init_validation(self, other):
        super(BaseDataLoader, self).__init__(sampler=other.valid_sampler,
                                             **other.init_kwargs)

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            dl = super(BaseDataLoader, type(self)).__new__(type(self))
            dl.init_validation(self)

            return dl
