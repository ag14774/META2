import logging
from abc import ABCMeta, abstractmethod

from model.util import all_tensors_to
from utils.util import get_global_rank


class BaseDeviceMapper(object, metaclass=ABCMeta):
    def __init__(self, n_gpu, ignore_parallelize=False):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.n_gpu = n_gpu

        self.logger.info("Initializing devices..")
        device, gpu_ids, n_gpu, n_processes = self.prepare_device()

        self.device = device
        self.gpu_ids = gpu_ids
        self.n_gpu = n_gpu
        self.n_processes = n_processes
        self.ignore_parallelize = ignore_parallelize

        if get_global_rank() == 0:
            self.logger.info(
                f"Number of running processes: {self.n_processes}")
            self.logger.info(f"Number of usable GPUs: {self.n_gpu}")

    @abstractmethod
    def prepare_device(self):
        raise NotImplementedError

    @abstractmethod
    def parallelize_model(self, model):
        raise NotImplementedError

    def map_modules(self, modules):
        return all_tensors_to(modules, device=self.device)

    def get_master_device(self):
        return self.device
