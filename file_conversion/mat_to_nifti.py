import os 
import argparse
import hdf5storage
import shutil
import SimpleITK as sitk
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
data_dir = os.path.join(args.path, 'data')

labels = []
images = []
masks = []
os.makedirs(os.path.join(args.path, 'output'), exist_ok=True)
os.makedirs(os.path.join(args.path, 'mask'), exist_ok=True)

for i, file in enumerate(tqdm(os.listdir(data_dir))):

    mat_file = hdf5storage.loadmat(os.path.join(data_dir, file))['cjdata'][0]

    image = mat_file[2]
    mask = mat_file[4]

    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask.astype(int))
    sitk.WriteImage(image, os.path.join(args.path, 'output', file.split('.')[0] + '.nii'))
    sitk.WriteImage(mask, os.path.join(args.path, 'mask', file.split('.')[0] + '.nii'))

shutil.rmtree(data_dir)

