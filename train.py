import datetime
from models import create_model
from data import create_dataset
from options.train_options import TrainingOptions
from tqdm import tqdm

if __name__ == '__main__':

    opt = TrainingOptions().parse()     # get all training options

    dataset = create_dataset(opt)
    model = create_model(opt)  # create model from opt.model
    model.setup(opt)
    model.train()

    for epoch in (range(0, opt.n_epochs)):
        for idx, batch in enumerate(tqdm(dataset.dataloader_train, desc='Epoch {}/{}'.format(epoch, opt.n_epochs))):
            input_img = batch['img']
            mask_img = batch['mask']

            model.set_input(input_img, mask_img)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()
