from music.custom_elements.riff.drum_riff import *
from music.process.audio_related import *


class Phrase(object):
    def __init__(self, length, bpm):
        self.length = length

        self.bpm = bpm

        self.pm = None

        self.save_dir = 'D:/PycharmProjects/RiffGAN//data/pieces/phrases/'
        self.midi_path = ''

    def save_midi(self, name):
        self.midi_path = self.save_dir + 'midi/' + name + '.mid'
        self.pm.write(self.midi_path)

    def play_it(self):
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music(self.midi_path)

    def play_with_no_init(self):
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music_without_init(self.midi_path)



