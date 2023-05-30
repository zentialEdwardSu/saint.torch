# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# activate =  [
#         nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
#         nn.Sigmoid, nn.Tanh,nn.SiLU,nn.GELU
# ]

class Clamp(nn.Module):
    """Clamp activation layer.
        Copyright (c) OpenMMLab.

    This activation function is to clamp the feature map value within
    :math:`[min, max]`. More details can be found in ``torch.clamp()``.

    Args:
        min (Number | optional): Lower-bound of the range to be clamped to.
            Default to -1.
        max (Number | optional): Upper-bound of the range to be clamped to.
            Default to 1.
    """

    def __init__(self, min: float = -1., max: float = 1.):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Clamped tensor.
        """
        return torch.clamp(x, min=self.min, max=self.max)

def build_activation_layer(cfg: Dict,inplace=True) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - custom_scope(str): e.g. 'mynorm.myfloder.norm'
            - layer args: these will be unpack as **kwargs to init activation layer.
        inplace(bool) handy args to control inplace args in activation layer
    Note:
        ReLU, LeakyReLU, PReLU, RReLU, ReLU6, ELU,
        Sigmoid, Tanh,SiLU,GELU

    Returns:
        nn.Module: Created activation layer.
    """
    if not isinstance(cfg,dict):
        raise TypeError(f"except cfg as a dict but get {type(cfg)}")
    if 'type' not in cfg:
        raise KeyError(f"'type' is necessary in cfg")

    _cfg = cfg.copy()
    _cfg.setdefault("inplace",inplace) # inplace will be place into cfg
    active_type = _cfg.pop("type")
    custom_scope = cfg.pop("custom_scope")
    if custom_scope:
        active_layer = __import__(custom_scope,fromlist=[type]) # import custom nb normlization layer from specific scope
    elif active_type == "Clamp":
        active_layer = Clamp
    else:
        active_layer = getattr(nn,active_type) # import from torch.nn

    if not active_type:
        raise ValueError(f"norm layer with type {active_type} can't be found in f{ custom_scope if custom_scope else 'nn'}")
    
    layer = active_layer(**cfg)

    return layer

    
