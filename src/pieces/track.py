from src.custom_elements.riff import *
from src.custom_elements.toolkit import *
import pretty_midi


class Track:
    def __init__(self, name, root_note_name, bpm, mode_change=False, metre_change=False):
        self.name = name
        self.riff_list = []

        if mode_change is True:
            assert type(mode_change) is list
        if metre_change is True:
            assert type(bpm) is list
        self.mode_change = mode_change
        self.bpm = bpm
        # (no, start_bar)

        self.pm = pretty_midi.PrettyMIDI()

    def add_riff(self, riff):
        assert type(riff) is GuitarRiff
        self.riff_list.append(riff)

    def add_riffs(self, riff_list):
        self.riff_list += riff_list

    def get_riff_num(self):
        print(len(self.riff_list))

    def save(self, save_path):
        self.pm.write(save_path)
