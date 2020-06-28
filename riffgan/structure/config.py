class Config(object):
    def __init__(self):

        ##########################
        # Info

        self.name = 'riff_gan'
        # self.name = 'SMGT'

        self.dataset_name = 'grunge_library'
        self.instr_type = 'guitar'

        self.network_name = 'riff_net'
        # self.network_name = 'midinet'

        self.time_step = 64
        self.bar_length = 4
        self.note_valid_range = (24, 108)
        self.note_valid_length = 84

        self.phase = 'train'
        self.continue_train = False

        ###########################

        ###########################
        # Structure

        self.model = 'base'  # three different models, base, partial, full

        self.use_image_pool = True
        self.image_pool_info = 'pooled' if self.use_image_pool else 'not_pooled'
        self.image_pool_max_size = 20

        self.bat_unit_eta = 0.2

        ##########################

        ##########################
        # Train

        self.gaussian_std = 1
        self.seed_size = 256

        self.sigma_c = 1.0
        self.sigma_d = 1.0

        self.gpu = True

        self.beta1 = 0.9  # Adam optimizer beta1 & 2
        self.beta2 = 0.999

        self.lr = 0.0002
        self.milestones = [2, 5, 8, 11, 13, 15, 17, 19, 20]
        self.gamma = 0.5

        self.weight_decay = 0.0

        self.no_flip = True
        self.num_threads = 0
        self.batch_size = 16
        self.max_epoch = 50
        self.epoch_step = 5

        # self.data_shape = (self.batch_size, 1, 64, 84)
        if self.instr_type == 'guitar':
            self.input_shape = (1, 64, 60)
        else:
            assert self.instr_type == 'bass'
            self.input_shape = (1, 64, 48)

        self.pitch_range = self.input_shape[2]

        self.plot_every = 100  # iterations
        self.save_every = 1  # epochs

        self.start_epoch = 0

        ##########################

        ##########################
        # Save Paths

        self.root_dir = 'd:/riff_gan'

        self.save_path = self.root_dir + '/checkpoints/' + '{}_{}_{}_gn{}_lr{}_wd{}'.format(
            self.name, self.model, self.image_pool_info,
            self.gaussian_std, self.lr, self.weight_decay)

        self.model_path = self.save_path + '/models'
        self.checkpoint_path = self.save_path + '/checkpoints'

        self.log_path = self.save_path + '/info.log'
        self.loss_save_path = self.save_path + '/losses.json'

        self.test_path = self.save_path + '/test_results'
        # self.test_save_path = self.test_path + '/' + self.direction

        self.G_save_path = self.model_path + '/G/'
        self.D_save_path = self.model_path + '/D/'

        ##########################
