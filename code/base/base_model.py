from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = self.parameters(return_groups=False,
                                           requires_grad=True)

        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(
            BaseModel,
            self).__str__() + '\nTrainable parameters: {}'.format(params)

    def parameters(self,
                   recurse=True,
                   return_groups=False,
                   requires_grad=False):
        if not return_groups:
            params = super().parameters(recurse=recurse)
            if requires_grad:
                params = filter(lambda p: p.requires_grad, params)
        else:
            params = super().parameters(recurse=recurse)
            if requires_grad:
                params = filter(lambda p: p.requires_grad, params)
        return params


class MoEModel(BaseModel):
    @abstractmethod
    def use_strategy(self, strategy):
        raise NotImplementedError


class TwoPartModel(BaseModel):
    @abstractmethod
    def get_partA(self):
        raise NotImplementedError

    @abstractmethod
    def get_partB(self):
        raise NotImplementedError
