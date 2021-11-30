import random 
import numpy as np
import torch.utils.data as data
from abc import ABC, abstractmethod 

class BaseDataset(data.Dataset, ABC):

    def __init__(self, opt):
        self.opt = opt 
        self.root = opt.dataroot

    @abstractmethod
    def __len__(self):
        return 0 

    @abstractmethod
    def __getitem__(self, index):
        pass