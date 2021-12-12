from .base_options import BaseOptions

class TrainingOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        

        self.isTrain=True

        return parser