from data.base_dataset import BaseDataset

class SingleDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

    def __getitem__(self, index):
        return None

    def __len__(self):

        return None