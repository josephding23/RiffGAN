import time
import re
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam
from riffgan.data.grunge_library import UnitRiffDataset
import torch.nn as nn
import shutil

from torchnet.meter import MovingAverageValueMeter

from riffgan.networks.riffnet_v2.discriminator_v2 import Discriminator as RiffD_v2
from riffgan.networks.riffnet_v2.generator_v2 import Generator as RiffG_v2

from riffgan.networks.riffnet_v1.discriminator_v1 import Discriminator as RiffD_v1
from riffgan.networks.riffnet_v1.generator_v1 import Generator as RiffG_v1

from riffgan.networks.midinet.discriminator import Discriminator as MidiNetD
from riffgan.networks.midinet.generator import Generator as MidiNetG


from riffgan.config import Config
from riffgan.structure.image_pool import ImagePool
from riffgan.structure.random_seed import *

import logging
import colorlog

from util.data_convert import *
from util.npy_related import *
from util.fix_generated_song import *
from riffgan.networks.midinet.utility import *

from music.custom_elements.drum_riff.auto_generate_drum import *


class RiffGAN(object):
    def __init__(self):
        torch.autograd.set_detect_anomaly(True)
        self.opt = Config()

        self.device = torch.device('cuda') if self.opt.gpu else torch.device('cpu')
        self.pool = ImagePool(self.opt.image_pool_max_size)

        self.logger = logging.getLogger()
        self.set_up_terminal_logger()

        self._build_model()

    def set_up_terminal_logger(self):
        self.logger.setLevel(logging.INFO)
        ch = colorlog.StreamHandler()
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(fg_cyan)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )

        ch.setFormatter(color_formatter)
        self.logger.addHandler(ch)

    def _build_model(self):

        if self.opt.network_name == 'riff_net_v1':
            self.generator = RiffG_v1(self.opt.pitch_range, self.opt.seed_size)
            self.discriminator = RiffD_v1(self.opt.pitch_range)

        elif self.opt.network_name == 'riff_net_v2':
            self.generator = RiffG_v2(self.opt.pitch_range, self.opt.seed_size)
            self.discriminator = RiffD_v2(self.opt.pitch_range)

        else:
            assert self.opt.network_name == 'midi_net'
            self.generator = MidiNetG(self.opt.pitch_range)
            self.discriminator = MidiNetD(self.opt.pitch_range)

        init_weight_(self.generator)
        init_weight_(self.discriminator)

        if self.opt.gpu:
            self.generator.to(self.device)
            # summary(self.generator, input_size=self.opt.input_shape)

            self.discriminator.to(self.device)
            # summary(self.discriminator, input_size=self.opt.input_shape)

        self.G_optimizer = Adam(params=self.generator.parameters(), lr=self.opt.g_lr,
                                betas=(self.opt.beta1, self.opt.beta2))
        self.D_optimizer = Adam(params=self.discriminator.parameters(), lr=self.opt.d_lr,
                                betas=(self.opt.beta1, self.opt.beta2))

        self.G_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer, T_0=2, T_mult=2, eta_min=4e-08)
        self.D_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.D_optimizer, T_0=2, T_mult=2, eta_min=4e-08)

    def find_latest_checkpoint(self):
        path = self.opt.D_save_path
        file_list = os.listdir(path)
        match_str = r'\d+'
        epoch_list = sorted([int(re.findall(match_str, file)[0]) for file in file_list])
        if len(epoch_list) == 0:
            raise Exception('No model to load.')
        latest_num = epoch_list[-1]
        return latest_num

    def continue_from_latest_checkpoint(self):
        latest_checked_epoch = self.find_latest_checkpoint()
        self.opt.start_epoch = latest_checked_epoch + 1

        G_filename = f'{self.opt.name}_G_{latest_checked_epoch}.pth'
        D_filename = f'{self.opt.name}_D_{latest_checked_epoch}.pth'

        G_path = self.opt.G_save_path + G_filename
        D_path = self.opt.D_save_path + D_filename

        self.generator.load_state_dict(torch.load(G_path))
        self.discriminator.load_state_dict(torch.load(D_path))

        print(f'Loaded model from epoch {self.opt.start_epoch-1}')

    def save_model(self, epoch):
        G_filename = f'{self.opt.name}_G_{epoch}.pth'
        D_filename = f'{self.opt.name}_D_{epoch}.pth'

        if epoch - self.opt.save_every >= 0:
            G_old_filename = f'{self.opt.name}_G_{epoch - self.opt.save_every}.pth'
            D_old_filename = f'{self.opt.name}_D_{epoch - self.opt.save_every}.pth'

            os.remove(os.path.join(self.opt.G_save_path, G_old_filename))
            os.remove(os.path.join(self.opt.D_save_path, D_old_filename))

        G_filepath = os.path.join(self.opt.G_save_path, G_filename)
        D_filepath = os.path.join(self.opt.D_save_path, D_filename)

        torch.save(self.generator.state_dict(), G_filepath)
        torch.save(self.discriminator.state_dict(), D_filepath)

        self.logger.info(f'model saved')

    def reset_save(self):
        if os.path.exists(self.opt.save_path):
            shutil.rmtree(self.opt.save_path)

        os.makedirs(self.opt.save_path, exist_ok=True)
        os.makedirs(self.opt.model_path, exist_ok=True)
        os.makedirs(self.opt.checkpoint_path, exist_ok=True)
        os.makedirs(self.opt.test_path, exist_ok=True)

        os.makedirs(self.opt.G_save_path, exist_ok=True)
        os.makedirs(self.opt.D_save_path, exist_ok=True)

    def train(self):
        torch.cuda.empty_cache()

        ######################
        # Save / Load model
        ######################

        if self.opt.continue_train:
            try:
                self.continue_from_latest_checkpoint()
            except Exception as e:
                self.logger.error(e)
                self.opt.continue_train = False
                self.reset_save()
        else:
            self.reset_save()

        dataset = UnitRiffDataset(self.opt.dataset_name, self.opt.instr_type)
        dataset_size = len(dataset)
        iter_num = int(dataset_size / self.opt.batch_size)

        self.logger.info(f'Dataset {self.opt.dataset_name} loaded, size {dataset_size}')

        ######################
        # Initiate
        ######################

        criterionGAN = nn.BCEWithLogitsLoss()

        GLoss_meter = MovingAverageValueMeter(self.opt.plot_every)
        DLoss_meter = MovingAverageValueMeter(self.opt.plot_every)

        losses = {}

        ######################
        # Start Training
        ######################

        for epoch in range(self.opt.max_epoch):
            loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=True,
                                num_workers=self.opt.num_threads, drop_last=False)
            epoch_start_time = time.time()

            for i, data in enumerate(loader):

                batch_size = data.size(0)
                # print(batch_size)

                real_label = torch.ones(size=[batch_size, 1], device=self.device)
                fake_label = torch.zeros(size=[batch_size, 1], device=self.device)

                seed = np.array([generate_random_seed(1, self.opt.instr_type, pattern=self.opt.chord_type)
                                for _ in range(batch_size)])
                # print(seed.shape)
                noise = torch.randn(batch_size, self.opt.seed_size, device=self.device)
                seed = torch.from_numpy(seed).to(device=self.device, dtype=torch.float)

                fake_data = self.generator(noise, seed, batch_size)
                D_fake = self.discriminator(fake_data, batch_size)

                real_data = torch.unsqueeze(data, 1).to(device=self.device, dtype=torch.float)
                D_real = self.discriminator(real_data, batch_size)
                # print(D_fake.shape)

                ######################
                # Generator
                ######################

                self.G_optimizer.zero_grad()
                loss_G = criterionGAN(D_fake, real_label)
                loss_G.backward(retain_graph=True)

                self.G_optimizer.step()

                self.G_optimizer.zero_grad()
                loss_G = criterionGAN(D_fake, real_label)
                loss_G.backward(retain_graph=True)

                self.G_optimizer.step()

                GLoss_meter.add(loss_G.item())

                ######################
                # Discriminator
                ######################

                self.D_optimizer.zero_grad()

                loss_D_real = criterionGAN(D_real, real_label)
                loss_D_fake = criterionGAN(D_fake, fake_label)

                loss_D = 0.5 * loss_D_real + 0.5 * loss_D_fake
                loss_D.backward()

                self.D_optimizer.step()
                DLoss_meter.add(loss_D.item())

            if epoch % self.opt.save_every == 0 or epoch == self.opt.max_epoch - 1:
                self.save_model(epoch)

            losses['loss_G'] = float(GLoss_meter.value()[0])
            losses['loss_D'] = float(DLoss_meter.value()[0])

            self.G_scheduler.step(epoch)
            self.D_scheduler.step(epoch)

            epoch_time = int(time.time() - epoch_start_time)

            self.logger.info(f'Epoch {epoch} finished, cost time {epoch_time}\n')
            self.logger.info(str(losses) + '\n\n')

    def test(self):
        from music.custom_elements.rhythm_riff.guitar_riff import GuitarRiff
        griff = GuitarRiff(measure_length=2,
                           degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                              ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                           time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
        griff.add_notes_to_pm(root_note_name='G2', bpm=120, instr=27)
        pm, shape = generate_nonzeros_from_pm(griff.pm, 120, 2, self.opt.instr_type)
        data = generate_sparse_matrix_from_nonzeros(pm, shape)
        # plot_data(database[0, :, :])

        torch.cuda.empty_cache()

        ######################
        # Save / Load model
        ######################

        self.continue_from_latest_checkpoint()

        for i in range(10):

            # noise = torch.unsqueeze(torch.from_numpy(database), 1).to(device=self.device, dtype=torch.float)

            random_riff = generate_random_seed(2, self.opt.instr_type, pattern=self.opt.chord_type)
            # plot_data(random_riff[0, :, :], self.opt.input_shape)
            noise = torch.randn(2, self.opt.seed_size, device=self.device)
            seed = torch.unsqueeze(torch.from_numpy(random_riff)
                                   , 1).to(device=self.device, dtype=torch.float)

            # noise = torch.unsqueeze(torch.from_numpy(data), 1).to(device=self.device, dtype=torch.float)
            # plot_data(noise[0, 0, :, :], shape=self.opt.input_shape)
            ori_sample = seed.cpu().detach().numpy()
            fake_sample = self.generator(noise, seed, 2).cpu().detach().numpy()
            # print(fake_sample[0, :, :])

            # plot_data(fake_sample[0, 0, :, :])

            save_midis(ori_sample, f'../data/generated_music/ori{str(i+1)}.mid', self.opt.instr_type)
            # merge_short_notes(f'../../data/generated_music/ori{str(i+1)}.mid', self.opt.instr_type)

            save_midis(fake_sample, f'../data/generated_music/gen{str(i+1)}.mid', self.opt.instr_type)
            merge_short_notes(f'../data/generated_music/gen{str(i+1)}.mid', self.opt.instr_type)


def reduce_mean(x):
    output = torch.mean(x, 0, keepdim=False)
    output = torch.mean(output, -1, keepdim=False)
    return output


if __name__ == '__main__':
    riff_gan = RiffGAN()
    riff_gan.test()