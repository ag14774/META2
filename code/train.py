import argparse
import collections
import random

import dataloader.dataloaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import dev_mapper.device_mapper as module_mapper
import numpy as np
import torch
import torch.nn.modules.loss as torch_loss
import trainer.optim as module_optim
import trainer.trainer as module_trainer
from parse_config import ConfigParser


def main(config):
    assert np  # Silence pyflake
    logger = config.get_logger('train')

    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    device_mapper = config.initialize('mapper', module_mapper)

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch)

    model = device_mapper.parallelize_model(model)
    logger.info(model)

    # get function handles of loss and metrics
    loss = config.initialize('loss', [torch_loss, module_loss])
    metrics = config.initialize('metrics', module_metric)

    # Move loss and metrics parameters to correct GPU
    loss = device_mapper.map_modules(loss)
    metrics = device_mapper.map_modules(metrics)

    # build optimizer, learning rate scheduler. delete every lines containing
    # lr_scheduler for disabling scheduler
    is_multiopt = config.config['optimizer']['type'] == 'MultiOpt'
    try:
        trainable_params = model.parameters(return_groups=is_multiopt,
                                            requires_grad=True)
    except TypeError:
        trainable_params = filter(lambda p: p.requires_grad,
                                  model.parameters())

    optimizer = config.initialize('optimizer', [torch.optim, module_optim],
                                  trainable_params)

    lr_scheduler = config.initialize('lr_scheduler',
                                     [torch.optim.lr_scheduler, module_optim],
                                     optimizer)

    trainer = config.initialize('trainer', module_trainer, model, loss,
                                metrics, optimizer, config, data_loader,
                                valid_data_loader, lr_scheduler,
                                device_mapper.get_master_device())

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c',
                      '--config',
                      default=None,
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-r',
                      '--resume',
                      default=None,
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d',
                      '--device',
                      default=None,
                      type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values
    # given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'],
                   type=float,
                   target=('optimizer', 'args', 'lr')),
        CustomArgs(['--bs', '--batch_size'],
                   type=int,
                   target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
