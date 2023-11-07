from torch import nn
from typing import Dict, Tuple, Union
from saint_core.utils import  logcat
from saint_core.utils.logcat import print_log

def build_norm_layer(cfg:Dict,num_features:int,postfix:Union[int,str]=''):
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:
            - type (str): Layer type.
            - custom_scope(str): e.g. 'mynorm.myfloder.norm'
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.
    Note:
        support type(nn): [BatchNorm1d,BatchNorm2d,BatchNorm3d,SyncBatchNorm\
            GroupNorm,LayerNorm,InstanceNorm1d,InstanceNorm2d,InstanceNorm3d]
    Return:
        name of created layer, layer ins)
        
    """

    # check function parameters
    if not isinstance(cfg,dict):
        raise TypeError(f"except cfg as a dict but get {type(cfg)}")
    if 'type' not in cfg:
        raise KeyError(f"'type' is necessary in cfg")

    _cfg = cfg.copy()
    _cfg.setdefault('eps', 1e-5)

    norm_layer_type = cfg.pop("type")
    custom_scope = cfg.pop("cuustom_scope")
    requires_grad = _cfg.pop('requires_grad', True)
    assert isinstance(postfix, (int, str))
    name = type + str(postfix)

    if custom_scope:
        norm_layer = __import__(custom_scope,fromlist=[type]) # import custom nb normlization layer from specific scope
    else:
        norm_layer = getattr(nn,norm_layer_type) # import from torch.nn

    if not norm_layer:
        raise ValueError(f"norm layer with type {norm_layer_type} can't be found in f{ custom_scope if custom_scope else 'nn'}")
    
    if norm_layer_type == 'GroupNorm':
        assert 'num_groups' in _cfg # when you use group normalization you should at least tell the numbers of group
        layer = norm_layer(num_channels=num_features, **_cfg)
    else:
        layer = norm_layer(num_features, **_cfg)
        if norm_layer_type == 'SyncBatchNorm' and hasattr(layer, '_specify_ddp_gpu_num'): # NOTE if unused del it later
            layer._specify_ddp_gpu_num(1)

    for param in layer.parameters():
        param.requires_grad = requires_grad # apply wheter gradiant setting

    return name, layer
    