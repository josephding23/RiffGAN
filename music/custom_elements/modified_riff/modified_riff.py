from music.custom_elements.rhythm_riff.riff import Riff
from music.custom_elements.rhythm_riff.guitar_riff import GuitarRiff
from music.custom_elements.rhythm_riff.bass_riff import BassRiff
from music.process.audio_related import play_music, play_music_without_init
from music.custom_elements.rhythm_riff.guitar_riff import parse_griff_json
from music.custom_elements.rhythm_riff.bass_riff import parse_briff_json
from util.music_generate import *
from util.data_plotting import *

import os
import pretty_midi


class ModifiedRiff:
    def __init__(self, original_riff, option):
        assert isinstance(original_riff, Riff)
        self.original_riff = original_riff
        self.option = option
        self.nonzeros = None
        self.shape = None

        self.save_dir = ''
        self.midi_path = ''
        self.fig_path = ''

        self.static_img_dir = '../web/static/img/modified_riffs/'

        self.pm = pretty_midi.PrettyMIDI()

    def __eq__(self, other):
        return self.original_riff == other.original_riff and self.option == other.option and \
               self.nonzeros == other.nonzeros and self.shape == other.shape

    def save_midi(self, name):
        # assert self.save_dir is not '' and os.path.exists(self.save_dir)
        self.midi_path = self.save_dir + 'midi/' + name + '.mid'
        self.pm.write(self.midi_path)

    def save_fig(self, name, riff_type):
        if riff_type == 'griff':
            instr_type = 'guitar'
        else:
            instr_type = 'bass'
        temp_midi_path = './temp.mid'
        self.pm.write(temp_midi_path)
        self.fig_path = self.static_img_dir + name + '.png'

        plot_midi_file(temp_midi_path, self.original_riff.measure_length, instr_type, save_image=True, save_path=self.fig_path)
        # os.remove(temp_midi_path)

    def add_notes_to_pm(self, instr):
        assert self.nonzeros is not None and self.shape is not None
        instr_track = pretty_midi.Instrument(program=instr)
        add_notes_from_nonzeros_to_instr(self.nonzeros, self.shape, instr_track)
        self.pm.instruments.append(instr_track)

    def set_nonzeros_and_shape(self, nonzeros, shape):
        self.nonzeros = nonzeros
        self.shape = shape

    def play_it(self):
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music(self.midi_path)

    def play_with_no_init(self):
        print(self.midi_path)
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music_without_init(self.midi_path)

    def export_json_dict(self):
        info_dict = {
            "original_riff": self.original_riff.export_json_dict(),
            'nonzeros': self.nonzeros,
            'shape': self.shape,
            "option": self.option,
            'modified': True
        }
        return info_dict


class ModifiedGuitarRiff(ModifiedRiff):
    def __init__(self, original_riff, option):
        ModifiedRiff.__init__(self, original_riff, option)
        assert isinstance(original_riff, GuitarRiff)
        self.save_dir = 'D:/PycharmProjects/RiffGAN/data/modified_riffs/griff/'


def parse_modified_griff_json(modified_riff_info):
    modified_riff = ModifiedGuitarRiff(
        original_riff=parse_griff_json(modified_riff_info['original_riff']),
        option=modified_riff_info['option']
    )
    modified_riff.set_nonzeros_and_shape(modified_riff_info['nonzeros'], modified_riff_info['shape'])
    return modified_riff


class ModifiedBassRiff(ModifiedRiff):
    def __init__(self, original_riff, option):
        ModifiedRiff.__init__(self, original_riff, option)
        assert isinstance(original_riff, BassRiff)
        self.save_dir = 'D:/PycharmProjects/RiffGAN/data/modified_riffs/briff/'


def parse_modified_briff_json(modified_riff_info):
    modified_riff = ModifiedBassRiff(
        original_riff=parse_griff_json(modified_riff_info['original_riff']),
        option=modified_riff_info['option']
    )
    modified_riff.set_nonzeros_and_shape(modified_riff_info['nonzeros'], modified_riff_info['shape'])
    return modified_riff
