import datetime
from models import create_model
from data import create_dataset
from options.train_options import TrainingOptions
import matplotlib.pyplot as plt


if __name__ == '__main__':

    opt = TrainingOptions().parse()     # get all training options

    dataset = create_dataset(opt)
    model = create_model(opt)           # create model from opt.model

    for idx, batch in enumerate(dataset.dataloader_train):
        input_img = batch['img']