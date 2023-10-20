"""
"""
import copy

import torch
from torch import nn

def extract_name_kwargs(obj):
    """
    """
    if isinstance(obj, dict):
        obj    = copy.copy(obj)
        name   = obj.pop('name')
        kwargs = obj
    else:
        name   = obj
        kwargs = {}

    return (name, kwargs)


def get_norm_layer3d(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm3d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm3d(features, **kwargs)

    raise ValueError(f"Unknown Layer: {name}")


def get_norm_layer2d(norm, features):
    name, kwargs = extract_name_kwargs(norm)

    if name is None:
        return nn.Identity(**kwargs)

    if name == 'layer':
        return nn.LayerNorm((features,), **kwargs)

    if name == 'batch':
        return nn.BatchNorm2d(features, **kwargs)

    if name == 'instance':
        return nn.InstanceNorm2d(features, **kwargs)

    raise ValueError(f"Unknown Layer: {name}")


def get_activ_layer(activ):
    name, kwargs = extract_name_kwargs(activ)

    if (name is None) or (name == 'linear'):
        return nn.Identity()

    if name == 'gelu':
        return nn.GELU(**kwargs)

    if name == 'relu':
        return nn.ReLU(**kwargs)

    if name == 'leakyrelu':
        return nn.LeakyReLU(**kwargs)

    if name == 'tanh':
        return nn.Tanh()

    if name == 'sigmoid':
        return nn.Sigmoid()

    raise ValueError(f"Unknown activation: {name}")


def get_jit_input(tensor, batch_size, device):
    """
    Get a dummy input for jit tracing
    """
    dummy = torch.ones_like(tensor)
    shape = (batch_size, ) + (1, ) * tensor.dim()
    dummy = dummy.repeat(shape)
    return dummy.to(device)


def get_lr(optim):
    """
    Get the current learning rate
    """
    for param_group in optim.param_groups:
        return param_group['lr']

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
