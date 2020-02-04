from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseMetric(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, output, target):
        raise NotImplementedError

    @abstractmethod
    def labels(self):
        '''
        Ordered list with metric names
        '''
        return [self.__class__.__name__]

    def __call__(self, *args, **kwargs):
        out = super().__call__(*args, **kwargs)
        try:
            assert iter(out)
            return out
        except TypeError:
            return [out]
