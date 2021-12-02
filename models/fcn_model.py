from .base_model import BaseModel
from .fcn.fcn_32 import FCN_32
from utils.colors_text import bcolors
import torch
from torchsummary import summary
import sys



class FCNModel(BaseModel):
    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.FCN = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        if opt.model == 'fcn_32':
            self.FCN = FCN_32().to(device)
        
        if not self.FCN:
            print("{}In folder FCN , there should be a file with class name that matches {}.{}".format(bcolors.FAIL, opt.model, bcolors.ENDC)) 
            print("{}Closing!!! {}".format(bcolors.FAIL, bcolors.ENDC))
            sys.exit()
        print(summary(self.FCN, (3, 250, 250)))