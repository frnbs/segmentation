import argparse
from matplotlib import pyplot as plt
import os
import SimpleITK as sitk

def show_images(args):
    mri_images = os.path.join(args.path, 'output')
    mask_images = os.path.join(args.path, 'mask')
    for img in os.listdir(mri_images):
        fig = plt.figure(figsize=(8, 8))
        columns = 1
        rows = 2

        fig.add_subplot(1, 2, 1)
        plt.imshow(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mri_images, img))))
        fig.add_subplot(1, 2, 2)
        plt.imshow(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_images,img))))
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="path of directory with the images and relative masks")
    args = parser.parse_args()

    show_images(args)