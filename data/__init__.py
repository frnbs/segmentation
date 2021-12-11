import importlib
import sys
import torch.utils.data
from torch.utils.data import random_split
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

        n_val = int(len(self.dataset) * 0.2)
        n_train = len(self.dataset) - n_val
        train_set, val_set = random_split(self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

        self.dataloader_train = torch.utils.data.DataLoader(
            train_set,
            batch_size=opt.batch_size,
            num_workers=int(opt.num_threads))
        self.dataloader_val = torch.utils.data.DataLoader(
            val_set,
            batch_size=opt.batch_size,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader_train):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data