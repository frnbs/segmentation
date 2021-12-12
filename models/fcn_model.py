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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
        if opt.model == 'fcn_32':
            self.model_names = ['FCN_32']
            self.FCN = FCN_32().to(device)

        if not self.FCN:
            print("{}In folder FCN , there should be a file with class name that matches {}.{}".format(bcolors.FAIL,
                                                                                                       opt.model,
                                                                                                       bcolors.ENDC))
            print("{}Closing!!! {}".format(bcolors.FAIL, bcolors.ENDC))
            sys.exit()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        input = torch.unsqueeze(input, 0)
        self.real_A = input.to(self.device)

    def forward(self):
        self.output = self.FCN(self.real_A.float())
        print("AAA")

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()

