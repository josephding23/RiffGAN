from music.db_fragments.music_fragment import MusicFragment
from util.npy_related import *


class Riff(MusicFragment):
    def __init__(self, path):
        MusicFragment.__init__(self, path)


class GuitarRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)


class BassRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)


class UnitRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)


class UnitGuitarRiff(UnitRiff):
    def __init__(self, path):
        UnitRiff.__init__(self, path)

    def save_nonzeros(self, save_path):
        nonzeros, shape = generate_nonzeros_from_pm(self.pm, bpm=120, length=1, instr_type='guitar')
        np.savez_compressed(save_path, nonzeros=nonzeros, shape=shape)


class UnitBassRiff(UnitRiff):
    def __init__(self, path):
        UnitRiff.__init__(self, path)

    def save_nonzeros(self, save_path):
        nonzeros, shape = generate_nonzeros_from_pm(self.pm, bpm=120, length=1, instr_type='bass')
        np.savez_compressed(save_path, nonzeros=nonzeros, shape=shape)
