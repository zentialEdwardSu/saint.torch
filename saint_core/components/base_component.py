import torch.nn as nn
import copy
from abc import ABCMeta
from typing import Iterable, List, Optional, Union,Dict

class BaseComponent(nn.Module,metaclass=ABCMeta):

    def __init__(self,init_cfg:Optional[Dict]=None):
        super.__init__()
        self.init_cfg = copy.deepcopy(init_cfg)

    def init_weights(self):
        pass

    def __repr__(self):
        s = super().__repr__()
        if self.init_cfg:
                s += f'\ninit_cfg={self.init_cfg}'
        return s