from torch import nn
from typing import Dict, Tuple, Optional

from saint_core.utils import  logcat
from saint_core.utils.logcat import print_log

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - custom_scope(str): e.g. 'mynorm.myfloder.norm'
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.
    Note: in nn we support [Conv1d,Conv2d,Conv3d]

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        _cfg = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        _cfg = cfg.copy()

    conv_layer_type = _cfg.pop('type')
    custom_scope = _cfg.pop('custom_scope')

    if custom_scope:
        conv_layer = __import__(custom_scope,fromlist=[type]) # import custom nb normlization layer from specific scope
    else:
        conv_layer = getattr(nn,conv_layer_type) # import from torch.nn

    if conv_layer is None:
        raise KeyError(f"norm layer with type {conv_layer_type} can't be found in f{ custom_scope if custom_scope else 'nn'}")
    layer = conv_layer(*args, **kwargs, **_cfg)

    return layer
