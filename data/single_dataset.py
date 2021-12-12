import os
import argparse
from data.base_dataset import BaseDataset
from pathlib import Path
from PIL import Image
import SimpleITK as sitk

class singledataset(BaseDataset):

    def __init__(self, opt : argparse):

        BaseDataset.__init__(self, opt)
        self.images_dir = Path(os.path.join(opt.dataroot, 'output'))
        self.mask_dir = Path(os.path.join(opt.dataroot, 'mask'))

        self.ids = [file.split('.')[0] for file in os.listdir(self.images_dir) if file.endswith('.nii')]
        if not self.ids:
            raise RuntimeError(f'No input file found in ')

    @classmethod
    def load(cls, filename):
        for file in filename:
            file_reader = sitk.ReadImage(str(file))
        return sitk.GetArrayFromImage(file_reader)

    def __getitem__(self, index):
        name = self.ids[index]
        mask_file = self.mask_dir.glob(name+ '.*')
        img_file = self.images_dir.glob(name+ '.*')

        mask = self.load(mask_file)
        img = self.load(img_file)
        return {"img": img, "mask": mask}

    def __len__(self):

        return len(self.ids)