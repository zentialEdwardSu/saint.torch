from typing import Dict

import torch.nn as nn

def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - custom_scope(str): custom model scope
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    custom_scope = cfg_.pop('custom_scope')

    if custom_scope:
        padding_layer = __import__(custom_scope,fromlist=[type]) # import custom nb normlization layer from specific scope
    else:
        padding_layer = getattr(nn,padding_type) # import from torch.nn

    if padding_layer is None:
        raise KeyError(f"norm layer with type {padding_type} can't be found in f{ custom_scope if custom_scope else 'nn'}")
    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer