import argparse 
import os 
import torch

class BaseOptions():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot',       required = True,                help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--model',          required = True,    type=str,   help='chooses which model to use. [fcn_32 | ]')
        parser.add_argument('--dataset_mode',   required = True,    type=str,   help='chooses how datasets are loaded. [single | ]')

        self.initialized = True 
        return parser 

    def gather_options(self):

        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt = parser.parse_args()
        
        model_name = opt.model 
        return parser.parse_args()


    def parse(self):
        
        opt = self.gather_options()
        self.opt = opt
        return self.opt