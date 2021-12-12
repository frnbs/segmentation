from .base_model import BaseModel
from .fcn.fcn_32 import FCN_32
from utils.colors_text import bcolors
import torch
from torch import optim
import sys
from torch import nn
import torch.nn.functional as F
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt
import cv2 as cv

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

    def set_input(self, input, mask):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        input_img = torch.unsqueeze(input, 0)
        mask = torch.unsqueeze(mask, 0)
        self.img = input_img.to(self.device)
        self.mask = mask.to(self.device)

    def forward(self):
        self.mask_pred = self.FCN(self.img.float())

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward

        criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.FCN.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        self.optimizer.zero_grad()
        self.forward()
        self.loss = criterion(self.mask_pred, self.mask.float()) + dice_loss(F.softmax(self.mask_pred, dim=1).float(), self.mask.float())
        self.loss.backward()
        self.optimizer.step()
