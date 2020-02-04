from abc import ABCMeta, abstractmethod

import numpy as np
import torch
# try:
#     from apex import amp
# except Exception:
#     from utils.dummy_amp import DummyAmp as amp
from logger.visualization import WriterTensorboardX
from numpy import inf
from utils.util import get_global_rank


class BaseTrainer(object, metaclass=ABCMeta):
    """
    Base class for all trainers
    """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler, main_device):
        self._set_defaults(model, loss, metrics, optimizer, config,
                           data_loader, valid_data_loader, lr_scheduler,
                           main_device)
        cfg_trainer = config['trainer']['extra_args']

        self.checkpoint_dir = config.save_dir
        if get_global_rank() == 0:
            # setup visualization writer instance
            enable_board = cfg_trainer['tensorboardX']
        else:
            enable_board = False
        self.writer = WriterTensorboardX(config.log_dir, self.logger,
                                         enable_board)

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def _set_defaults(self, model, loss, metrics, optimizer, config,
                      data_loader, valid_data_loader, lr_scheduler,
                      main_device):
        cfg_trainer = config['trainer']['extra_args']
        self.config = config
        self.logger = config.get_logger('trainer', cfg_trainer['verbosity'])

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.metrics_labels = [l for mtr in self.metrics for l in mtr.labels()]
        self.optimizer = optimizer

        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler

        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.keep_last = cfg_trainer.get('keep_last', 30)
        self.disable_hist = cfg_trainer.get('disable_hist', False)
        self.strict_load = cfg_trainer.get('strict_load', True)
        self.input_to_master = cfg_trainer.get('input_to_master', True)

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.lr_scheduler = lr_scheduler
        self.main_device = main_device

        self.log_step = int(np.sqrt(self.data_loader.batch_size))
        while self.log_step > len(self.data_loader):
            log_step = self.log_step // 8
            if log_step == 0:
                log_step = self.log_step // 2
            if log_step != 0:
                self.log_step = log_step
        self.start_epoch = 1
        self.not_improved_count = 0
        self.improved_since_last_save = False

    @abstractmethod
    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def valid_epoch(self, epoch):
        """
        Validation logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        self.not_improved_count = 0
        self.improved_since_last_save = False
        for epoch in range(self.start_epoch, self.epochs + 1):
            print()
            self.data_loader.step(epoch)

            result = self.train_epoch(epoch)
            if self.do_validation:
                self.valid_data_loader.step(epoch)
                val_log = self.valid_epoch(epoch)
                result = {**result, **val_log}

            if self.lr_scheduler is not None:
                self.logger.info(
                    f"Learning rate: {self.lr_scheduler.get_lr()}")
                self.lr_scheduler.step(epoch=epoch)

            if get_global_rank() == 0:
                log = self._log_info(result, epoch)
                early_stop = self._check_early_stop(log, epoch)

                if early_stop:
                    break

    def _eval_metrics(self, output, target):
        acc_metrics = torch.zeros(len(self.metrics_labels))
        # print("acc_metrics", acc_metrics.device)
        i = 0
        for metric in self.metrics:
            metric_out = metric(output, target)
            for res in metric_out:
                acc_metrics[i] += res
                self.writer.add_scalar('{}'.format(self.metrics_labels[i]),
                                       acc_metrics[i])
                i += 1
        return acc_metrics

    def _log_info(self, result, epoch):
        # save logged informations into log dict
        log = {'epoch': epoch}
        for key, value in result.items():
            if key == 'metrics':
                log.update({
                    mtr_label: value[i]
                    for i, mtr_label in enumerate(self.metrics_labels)
                })
            elif key == 'val_metrics':
                log.update({
                    'val_' + mtr_label: value[i]
                    for i, mtr_label in enumerate(self.metrics_labels)
                })
            else:
                log[key] = value

        # print logged informations to the screen
        for key, value in log.items():
            self.logger.info('    {:15s}: {}'.format(str(key), value))

        return log

    def _check_early_stop(self, log, epoch):
        # evaluate model performance according to configured metric,
        # save best checkpoint as model_best
        if self.mnt_mode != 'off':
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                improved = (self.mnt_mode == 'min'
                            and log[self.mnt_metric] <= self.mnt_best) or (
                                self.mnt_mode == 'max'
                                and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. "
                    "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                self.mnt_mode = 'off'
                improved = False
                self.not_improved_count = 0

            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                self.improved_since_last_save = True
            else:
                self.not_improved_count += 1

            if self.not_improved_count > self.early_stop:
                self.logger.info(
                    "Validation performance didn\'t improve for {} epochs. "
                    "Training stops.".format(self.early_stop))
                return True

        if epoch % self.save_period == 0:
            self._save_checkpoint(epoch,
                                  save_best=self.improved_since_last_save)
            self.improved_since_last_save = False

        return False

    @abstractmethod
    def create_checkpoint_dict(self, epoch):
        arch = type(self.model).__name__
        if isinstance(self.model, torch.nn.DataParallel):
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

    @abstractmethod
    def load_state_dict(self, state_dict, strict=True):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.load_state_dict(state_dict, strict=strict)
        else:
            return self.model.load_state_dict(state_dict, strict=strict)

    def _attempt_to_fix_state_dict(self, state_dict, missing_keys,
                                   unexpected_keys):
        for m in missing_keys:
            for u in unexpected_keys:
                if m in u or u in m:
                    state_dict[m] = state_dict.pop(u)
                    break
        return state_dict

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint
            to 'model_best.pth'
        """
        state = self.create_checkpoint_dict(epoch)
        filename = str(self.checkpoint_dir /
                       'checkpoint-epoch{}.pth'.format(epoch))

        self.logger.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)

        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            self.logger.info("Saving current best: model_best.pth ...")
            torch.save(state, best_path)

        self._remove_checkpoints()

    def _remove_checkpoints(self):
        checkpoints = self.checkpoint_dir.parent.rglob("checkpoint-epoch*.pth")
        checkpoints = sorted(checkpoints, key=lambda f: f.stat().st_mtime)
        checkpoints = checkpoints[:-self.keep_last]
        for c in checkpoints:
            c.unlink()

        bests = self.checkpoint_dir.parent.rglob("model_best.pth")
        bests = sorted(bests, key=lambda f: f.stat().st_mtime)
        bests = bests[:-1]
        for c in bests:
            c.unlink()

        configs = self.checkpoint_dir.parent.rglob("config.json")
        configs = sorted(configs, key=lambda f: f.stat().st_mtime)
        configs = configs[:-1]
        for c in configs:
            c.unlink()

        for f in self.checkpoint_dir.parent.glob('*'):
            if f.is_dir() and len(list(f.glob('*'))) == 0:
                f.rmdir()

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)

        if self.strict_load:
            self.start_epoch = checkpoint['epoch'] + 1
            self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config.raw['arch']:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is "
                "different from that of checkpoint. This may yield "
                "an exception while state_dict is being loaded.")
        try:
            missing, unexpected = self.load_state_dict(
                checkpoint['state_dict'], strict=self.strict_load)
            if not self.strict_load:
                self.logger.info(f"Missing layers: {missing}")
                self.logger.info(f"Unexpected layers: {unexpected}")
        except RuntimeError:
            checkpoint['state_dict'] = self._attempt_to_fix_state_dict(
                checkpoint['state_dict'], missing, unexpected)
            self.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when
        # optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config.raw[
                'optimizer']['type']:
            self.logger.warning(
                "Warning: Optimizer type given in config file is different "
                "from that of checkpoint. Optimizer parameters not resumed.")
        elif not self.strict_load:
            self.logger.warning("Warning: Strict loading is disabled. "
                                "Optimizer parameters not resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if checkpoint['config']['lr_scheduler']['type'] != self.config.raw[
                'lr_scheduler']['type']:
            self.logger.warning(
                "Warning: LR Scheduler type given in config file is different "
                "from that of checkpoint. Parameters not resumed.")
        elif not self.strict_load:
            self.logger.warning("Warning: Strict loading is disabled. "
                                "LR scheduler state not resumed.")
        else:
            if self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(
                self.start_epoch))
