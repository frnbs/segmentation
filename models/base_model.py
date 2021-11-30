import os 
import torch 
from abc import ABC 

class BaseModel(ABC):

    def __init__(self, opt):
        self.opt = opt 