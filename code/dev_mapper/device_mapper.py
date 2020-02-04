import os

import torch
import torch.cuda
import torch.distributed as dist
import torch.multiprocessing
from base.base_device_mapper import BaseDeviceMapper
from dev_mapper.utils import CustomDistributedDataParallel


class SingleNodeDataParallel(BaseDeviceMapper):
    def __init__(self, n_gpu, ignore_parallelize=False):
        super().__init__(n_gpu, ignore_parallelize)

    def prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        returns: main_gpu, list of gpus avaialble to the current process,
            how many GPUs available in total in all pytorch processes
            involved in training the model.
        """
        n_gpu_use = self.n_gpu
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine,"
                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, "
                "but only {} are available on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids, len(list_ids), 1

    def parallelize_model(self, model):
        if self.ignore_parallelize:
            return model
        model = model.to(self.device)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=self.gpu_ids)
        return model


class MultiNodeDataParallel(BaseDeviceMapper):
    def __init__(self, n_gpu, ignore_parallelize=False):
        super().__init__(n_gpu, ignore_parallelize)

    def prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        torch.multiprocessing.set_start_method('spawn')
        n_gpu_use = self.n_gpu

        # Perform handshake
        if not dist.is_initialized():
            dist.init_process_group(backend='mpi')

        world_size = os.getenv("WORLD_SIZE")
        local_size = os.getenv("LOCAL_SIZE")
        n_nodes = os.getenv("NNODES")

        if not world_size or not local_size or not n_nodes:
            raise RuntimeError(
                ("Error: Environment variables are not set. "
                 "Are you sure you are running in a distributed environment?"))

        world_size = int(world_size)
        assert world_size == dist.get_world_size()

        local_size = int(local_size)
        rank = dist.get_rank()
        n_nodes = int(n_nodes)
        local_rank = rank % local_size

        n_gpu = torch.cuda.device_count() * n_nodes
        if n_gpu == 0 and n_gpu_use > 0:
            self.logger.warning(
                ("Warning: There\'s no GPU available in this machine. "
                 "Using CPU for training.."))
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                ("Warning: The number of GPUs configured to use is {},"
                 " but {} are available.").format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if n_gpu_use > 0 and n_gpu_use != world_size:
            raise RuntimeError(
                ("Error: The number of GPUs configured to use is {},"
                 " but {} processes have been spawned.").format(
                     n_gpu_use, world_size))

        if n_gpu:
            device = torch.device(f'cuda:{local_rank}')
            list_ids = [device]
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
            list_ids = []

        return device, list_ids, n_gpu_use, dist.get_world_size()

    def parallelize_model(self, model):
        if self.ignore_parallelize:
            return model
        model = model.to(self.device)
        if self.n_processes > 1:
            if self.n_gpu > 1:
                assert self.n_gpu == self.n_processes
                model = CustomDistributedDataParallel(
                    model, device_ids=self.gpu_ids, output_device=self.device)
            if self.n_gpu == 0:
                model = torch.nn.parallel.DistributedDataParallelCPU(model)

        return model


class MultiNodeDataParallelNCCL(MultiNodeDataParallel):
    def __init__(self, n_gpu, ignore_parallelize=False):
        super().__init__(n_gpu, ignore_parallelize)

    def prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        device, gpu_list, n_gpu_use, n_proc = super().prepare_device()
        hostname = os.getenv('HOSTNAME')
        hostname = [ord(x) for x in hostname]
        max_size = 128
        hostname = torch.tensor(hostname, device=device).float()
        hostname = torch.nn.functional.pad(hostname,
                                           (0, max_size - len(hostname)))

        tensor_list = [
            torch.zeros(max_size, device=device) for _ in range(n_proc)
        ]
        dist.all_gather(tensor_list, hostname)
        master = tensor_list[0]
        chars = [chr(x) for x in master if x != 0]
        master = "".join(chars)
        os.environ["MASTER_ADDR"] = master
        os.environ["MASTER_PORT"] = str(29500)
        os.environ["RANK"] = str(dist.get_rank())
        os.environ["LOCAL_RANK"] = str(dist.get_rank() %
                                       int(os.getenv("LOCAL_SIZE")))
        dist.destroy_process_group()
        dist.init_process_group(backend='nccl')
        return device, gpu_list, n_gpu_use, dist.get_world_size()
