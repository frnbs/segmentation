from data.base_dataset import BaseDataset
from pathlib import Path

class BasicDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)
        self.images_dir = Path(opt.images_dir)
        self.mask_dir = Path(opt.masks_dir)

    def __getitem__(self, index):
        return None

    def __len__(self):

        return len(self.ids)