import torch 
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, n_class=21):
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)