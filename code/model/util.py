import importlib
import math

import model.module as modules
import numpy as np
import torch
import torch.nn as nn
from logger.visualization import WriterTensorboardX


def import_matplotlib():
    global mpl
    global plt
    mpl = importlib.import_module('matplotlib')
    # mpl.use('Agg')
    plt = importlib.import_module('matplotlib.pyplot')


import_matplotlib()


def check_parameter_number(model, return_groups, requires_grad):
    res = model.parameters(return_groups=return_groups,
                           requires_grad=requires_grad)
    correct_params = model.parameters(return_groups=False,
                                      requires_grad=requires_grad)
    correct_params = sum([np.prod(p.size()) for p in correct_params])

    actual_params = 0
    for k, v in res.items():
        temp = sum([np.prod(p.size()) for p in v])
        actual_params += temp

    assert actual_params == correct_params, (
        f"actual_params: {actual_params}, correct_params: {correct_params}")


def convert_batch_to_counts(num_of_classes, batch_labels):
    res_vector = torch.bincount(batch_labels, minlength=num_of_classes).float()
    res_vector = res_vector.view(1, -1)
    return res_vector


def convert_batch_to_prob(num_of_classes, batch_labels, eps=0.01):
    res_vector = torch.bincount(batch_labels, minlength=num_of_classes).float()
    res_vector = res_vector + eps
    res_vector = res_vector / torch.sum(res_vector)
    res_vector = res_vector.view(1, -1)
    return res_vector


def convert_batch_to_log_prob(num_of_classes, batch_labels, eps=0.01):
    res_vector = torch.bincount(batch_labels, minlength=num_of_classes).float()
    res_vector = res_vector + eps
    res_vector = torch.log(res_vector) - torch.log(torch.sum(res_vector))
    res_vector = res_vector.view(1, -1)
    return res_vector


def write_prob_dist(output, target, label):
    output = output.view(-1)
    target = target.view(-1)
    writer = WriterTensorboardX(None, None, None)  # Singleton
    if writer.mode == 'train':
        return
    if writer.step % 100 != 0:
        return
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(len(output)), output.cpu(), alpha=0.6)
    ax.plot(np.arange(len(target)), target.cpu(), alpha=0.6)
    ax.legend(['Output', 'Target'], loc=2)
    ax.set_xlabel('Taxon ID', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(f"Distribution of taxons at the {label} rank")
    writer.add_figure(label, fig)


def get_all_experts(model):
    return get_all_instances(model, modules.Expert)


def get_all_instances(model, cls):
    return [m for m in model.modules() if isinstance(m, cls)]


def get_a_tensor(input, exclude_empty=False):
    '''
    Attempts to iterate a given input datastructure and finds the first
    instance of torch.Tensor that it finds
    '''
    if isinstance(input, torch.Tensor):
        if not exclude_empty or input.size(0) > 0:
            return input

    if not isinstance(input, list) and not isinstance(
            input, dict) and not isinstance(input, tuple):
        raise TypeError('Cannot find tensor in input..')

    if isinstance(input, dict):
        input = input.items()

    for i in input:
        try:
            return get_a_tensor(i, exclude_empty=exclude_empty)
        except TypeError:
            if i is input[-1]:
                raise TypeError('Cannot find tensor in input..')


def collect_tensors(input):
    if isinstance(input, torch.Tensor):
        return [input]

    if not isinstance(input, list) and not isinstance(
            input, dict) and not isinstance(input, tuple):
        return []

    if isinstance(input, dict):
        input = input.items()

    res = []
    for i in input:
        try:
            res += collect_tensors(i)
        except TypeError:
            pass
    return res


def all_tensors_to(input, device, non_blocking=False):
    try:
        return input.to(device=device, non_blocking=non_blocking)
    except AttributeError:
        pass

    if not isinstance(input, list) and not isinstance(
            input, dict) and not isinstance(input, tuple):
        return input

    is_dict = False
    if isinstance(input, dict):
        is_dict = True
        input = input.items()

    res = []
    for i in input:
        try:
            res.append(all_tensors_to(i, device, non_blocking=non_blocking))
        except TypeError:
            pass
    if is_dict:
        res = dict(res)
    elif isinstance(input, tuple):
        res = tuple(res)
    return res


def is_list_of_tensors(input):
    for i in input:
        if not torch.is_tensor(i):
            return False
    return True


def infer_device(input):
    tensors = collect_tensors(input)
    device = tensors[0].device
    for t in tensors:
        if t.device != device:
            raise TypeError('Cannot infer device from input..')
    return device


def normal_cdf(x, stddev):
    """Evaluates the CDF of the normal distribution.
    Normal distribution with mean 0 and standard deviation stddev,
    evaluated at x=x.
    input and output `Tensor`s have matching shapes.
    Args:
        x: a `Tensor`
        stddev: a `Tensor` with the same shape as `x`.
     Returns:
        a `Tensor` with the same shape as `x`.
    """
    return 0.5 * (1.0 + torch.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def batch_normalization(inplanes, adaptive=True, track_running_stats=True):
    if adaptive:
        return modules.AdaptiveBatchNorm2d(
            inplanes,
            momentum=0.99,
            eps=0.001,
            track_running_stats=track_running_stats)
    else:
        return nn.BatchNorm2d(inplanes,
                              momentum=0.99,
                              eps=0.001,
                              track_running_stats=track_running_stats)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=True)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=True)


def avgpool2x1(stride=(2, 1)):
    return nn.AvgPool2d(kernel_size=(2, 1), stride=stride)


def conv_h_w(h, w, in_planes, out_planes, stride=1, bias=True, padding='same'):
    if padding == 'valid':
        stride = tuple([max(1, i) for i in stride])
        return nn.Conv2d(in_planes,
                         out_planes,
                         kernel_size=(h, w),
                         stride=stride,
                         bias=bias)
    elif padding == 'same':
        padding_h = (h - 1) // 2
        remainder_h = (h - 1) % 2
        padding_w = (w - 1) // 2
        remainder_w = (w - 1) % 2
        if isinstance(stride, int):
            stride = (stride, stride)
        stride_h, stride_w = stride
        if stride_h == 0:
            padding_h = 0
            remainder_h = 0
            stride_h = 1
        if stride_w == 0:
            padding_w = 0
            remainder_w = 0
            stride_w = 1
        return nn.Sequential(
            nn.ConstantPad2d((padding_w, padding_w + remainder_w, padding_h,
                              padding_h + remainder_h), 0),
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=(h, w),
                      stride=(stride_h, stride_w),
                      bias=bias))

    else:
        raise ValueError(
            f'padding must be either "same" or "valid". Got {padding}')
