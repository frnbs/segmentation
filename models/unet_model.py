from .base_model import BaseModel
from .Unet.Unet import Unet
from utils.colors_text import bcolors
import torch
from torchsummary import summary
import sys



class UNETModel(BaseModel):
    def __init__(self, opt):

        BaseModel.__init__(self, opt)
        self.UNET = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        if opt.model == 'unet':
            self.UNET = Unet().to(device)
        
        if not self.FCN:
            print("{}In folder UNET , there should be a file with class name that matches {}.{}".format(bcolors.FAIL, opt.model, bcolors.ENDC)) 
            print("{}Closing!!! {}".format(bcolors.FAIL, bcolors.ENDC))
            sys.exit()
        print(summary(self.UNET, (3, 250, 250)))