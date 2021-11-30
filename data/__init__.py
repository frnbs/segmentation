import importlib
import sys
import torch.utils.data 
from data.base_dataset import BaseDataset
from utils.colors_text import bcolors

def find_dataset_using_name(dataset_name):

    dataset_filename = "data." + dataset_name + "_dataset"
    try:
        datasetlib = importlib.import_module(dataset_filename)
    except:
        print("{}Cannot find file {}.py{}".format(bcolors.FAIL, dataset_filename, bcolors.ENDC))
        print("{}Closing!!! {}".format(bcolors.FAIL, bcolors.ENDC))
        sys.exit()
        
    dataset = None 
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def create_dataset(opt):

    data_loader = CustomDatastDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset

class CustomDatastDataLoader():

    def __init__(self, opt):
        self.opt = opt 
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [{}] was created".format(type(self.dataset).__name__))

    def load_data(self):
        return self
