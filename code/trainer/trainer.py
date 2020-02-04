import matplotlib.pyplot as plt
import numpy as np
import torch
from base.base_trainer import BaseTrainer
from matplotlib.lines import Line2D
from torchvision.utils import make_grid
from model.util import all_tensors_to


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            print(n, ave_grads[-1], max_grads[-1])
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)
    ], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def make_grid2(tensor,
               nrow=8,
               padding=2,
               normalize=False,
               range=None,
               scale_each=False,
               pad_value=0):
    while len(tensor.shape) < 4:
        tensor = torch.unsqueeze(tensor, -1)
    return make_grid(tensor, nrow, padding, normalize, range, scale_each,
                     pad_value)


def memory_stats(logger):
    logger.info(f'Memory allocated: {torch.cuda.memory_allocated()}')
    logger.info(f'Max memory allocated: {torch.cuda.max_memory_allocated()}')
    logger.info(f'Memory cached: {torch.cuda.memory_cached()}')
    logger.info(f'Max memory allocated: {torch.cuda.max_memory_cached()}')


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler, main_device):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, config,
                                      data_loader, valid_data_loader,
                                      lr_scheduler, main_device)

    def create_checkpoint_dict(self, epoch):
        return super().create_checkpoint_dict(epoch)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)

    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = torch.zeros(len(self.metrics_labels))

        for batch_idx, (data, target) in enumerate(self.data_loader):
            target = all_tensors_to(target,
                                    self.main_device,
                                    non_blocking=True)
            if self.input_to_master:
                data = data.to(self.main_device, non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) +
                                 batch_idx)
            self.writer.add_scalar('loss', loss.item())

            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss.item()))
                # self.writer.add_image(
                #     'input', make_grid2(data.cpu(), nrow=8, normalize=True))

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        return log

    def valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()

        total_val_loss = 0
        total_val_metrics = torch.zeros(len(self.metrics_labels))

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                target = all_tensors_to(target,
                                        self.main_device,
                                        non_blocking=True)
                if self.input_to_master:
                    data = data.to(self.main_device, non_blocking=True)

                output = self.model(data)
                loss = self.loss(output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx,
                    'valid')
                self.writer.add_scalar('loss', loss.item())

                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

                # self.writer.add_image(
                #     'input', make_grid2(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        if not self.disable_hist:
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss':
            total_val_loss / len(self.valid_data_loader),
            'val_metrics':
            (total_val_metrics / len(self.valid_data_loader)).tolist()
        }


class DistTrainer(Trainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler, main_device):
        super(DistTrainer,
              self).__init__(model, loss, metrics, optimizer, config,
                             data_loader, valid_data_loader, lr_scheduler,
                             main_device)
        # TODO: BROADCAST PARAMETERS HERE TO CHECK IF THEY ARE THE SAME

    def create_checkpoint_dict(self, epoch):
        arch = type(self.model).__name__
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        lr_sched_dict = None
        if self.lr_scheduler:
            lr_sched_dict = self.lr_scheduler.state_dict()
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': lr_sched_dict,
            'monitor_best': self.mnt_best,
            'config': self.config.raw
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module.load_state_dict(state_dict, strict=strict)
        else:
            return self.model.load_state_dict(state_dict, strict=strict)

    def train_epoch(self, epoch):
        return super().train_epoch(epoch)

    def valid_epoch(self, epoch):
        return super().valid_epoch(epoch)
