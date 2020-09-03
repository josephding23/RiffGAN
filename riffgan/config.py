class Config(object):
    def __init__(self, network, dataset, instr_type, continue_training):

        assert network in ['midinet', 'riff_net_v1', 'riff_net_v2', 'riff_net_v3', 'riff_net_v4']
        assert dataset in ['grunge_library', 'jimi_library']
        assert instr_type in ['guitar', 'bass']

        self.network_name = network
        self.dataset_name = dataset
        self.instr_type = instr_type
        self.continue_train = continue_training

        ##########################
        # Info

        self.name = 'riff_gan'
        # self.name = 'SMGT'

        if self.instr_type == 'guitar':
            self.input_shape = (1, 64, 60)
            self.chord_type = '5'
        else:
            assert self.instr_type == 'bass'
            self.input_shape = (1, 64, 48)
            self.chord_type = ''

        self.pitch_range = self.input_shape[2]

        self.time_step = 64
        self.bar_length = 4
        self.note_valid_range = (24, 108)
        self.note_valid_length = 84

        self.phase = 'train'

        ###########################

        ###########################
        # Structure

        # self.model = 'base'  # three different models, base, partial, full

        self.use_image_pool = True
        self.image_pool_info = 'pooled' if self.use_image_pool else 'not_pooled'
        self.image_pool_max_size = 20

        self.bat_unit_eta = 0.2

        ##########################

        ##########################
        # Train

        self.gaussian_std = 1
        self.seed_size = 100
        self.unit_length = 1

        self.sigma_c = 1.0
        self.sigma_d = 1.0

        self.gpu = True

        self.beta1 = 0.5  # Adam optimizer beta1 & 2
        self.beta2 = 0.999

        self.batch_size = 32

        self.g_lr = 0.0002
        self.d_lr = 0.0001
        self.gamma = 0.5

        self.weight_decay = 0.01

        self.no_flip = True
        self.num_threads = 0
        self.max_epoch = 200
        self.epoch_step = 5

        self.plot_every = 100  # iterations
        self.save_every = 1  # epochs

        self.start_epoch = 0

        ##########################

        ##########################
        # Save Paths

        self.root_dir = '..'

        self.save_path = self.root_dir + '/checkpoints/' + '{}_{}_{}_{}'.format(
            self.name, self.dataset_name, self.network_name, self.instr_type)

        self.model_path = self.save_path + '/models'
        self.checkpoint_path = self.save_path + '/checkpoints'

        self.log_path = self.save_path + '/info.log'
        self.loss_save_path = self.save_path + '/losses.json'

        self.test_path = self.save_path + '/test_results'
        # self.test_save_path = self.test_path + '/' + self.direction

        self.G_save_path = self.model_path + '/G/'
        self.D_save_path = self.model_path + '/D/'

        ##########################
