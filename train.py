import datetime
import os 
import random 
import time 

import numpy as np
import torchvision.transforms as standard_transforms
from models import create_model
from data import create_dataset
from options.train_options import TrainingOptions


if __name__ == '__main__':

    opt = TrainingOptions().parse()     # get all training options

    dataset = create_dataset(opt)       # create dataset
    model = create_model(opt)           # create model from opt.model
