import pretty_midi
import math
import numpy as np
import torch

from music.custom_elements.rhythm_riff.riff import Riff
from music.custom_elements.rhythm_riff.bass_riff import BassRiff
from music.custom_elements.rhythm_riff.guitar_riff import GuitarRiff
from music.custom_elements.modified_riff.modified_riff import ModifiedRiff, ModifiedBassRiff, ModifiedGuitarRiff

from riffgan.riffgan_model import RiffGAN
from util.music_generate import *


def get_measure_length(bpm):
    return 60 / bpm * 4


def plot_data(data, shape):
    import matplotlib.pyplot as plt
    sample_data = data
    dataX = []
    dataY = []
    for time in range(shape[1]):
        for pitch in range(shape[2]):
            if sample_data[time][pitch] > 0.1:
                dataX.append(time)
                dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()


def generate_data_from_midi(path, measure_num, instr_type, bpm=120):
    pm = pretty_midi.PrettyMIDI(path)
    if instr_type == 'guitar':
        note_range = (36, 96)
        # standard tune: [E2, D6] -> [C2, C7)
    else:
        assert instr_type == 'bass'
        note_range = (24, 72)
        # standard tune: [E1, G4] -> [C1, C5)
    data = np.zeros(shape=(measure_num, 64, note_range[1]-note_range[0]), dtype=np.float)

    # data = np.zeros((segment_num, 64, 84), np.bool_)
    sixtyfourth_length = 60 / bpm / 16
    for instr in pm.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                start = int(round(note.start / sixtyfourth_length))
                end = int(round(note.end / sixtyfourth_length))
                pitch = note.pitch
                if pitch < note_range[0] or pitch >= note_range[1]:
                    continue
                else:
                    pitch -= note_range[0]
                    for time_raw in range(start, end):
                        segment = int(time_raw / 64)
                        time = time_raw % 64
                        data[(segment, time, pitch)] = 1.0

    return data


def modify_riff(riff, riff_type, no, option):
    if riff_type == 'griff':
        instr_type = 'guitar'
        modified_riff = ModifiedGuitarRiff(riff, option)
    else:
        assert riff_type == 'briff'
        instr_type = 'bass'
        modified_riff = ModifiedBassRiff(riff, option)

    assert isinstance(riff, Riff)
    riffgan = RiffGAN()
    if riff_type == 'griff':
        assert isinstance(riff, GuitarRiff)
    else:
        assert riff_type == 'briff'
        assert isinstance(riff, BassRiff)

    riffgan.opt.instr_type = instr_type

    if option == 'Jimify':
        riffgan.opt.dataset_name = 'jimi_library'
    else:
        assert option == 'Grungefy'
        riffgan.opt.dataset_name = 'grunge_library'

    riffgan.continue_from_latest_checkpoint()

    ori_midi_path = riff.midi_path
    ori_riff_data = generate_data_from_midi(ori_midi_path, riff.measure_length, instr_type)

    noise = torch.randn(2, riffgan.opt.seed_size, device=riffgan.device)
    seed = torch.unsqueeze(torch.from_numpy(ori_riff_data)
                           , 1).to(device=riffgan.device, dtype=torch.float)

    modified_riff.set_midi_path(f'temp_{riff_type}_{no}_{option}')

    fake_sample = riffgan.generator(noise, seed, riff.measure_length).cpu().detach().numpy()

    save_midis(fake_sample, modified_riff.midi_path, instr_type)

    return modified_riff.export_json_dict()
