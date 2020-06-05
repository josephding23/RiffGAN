from src.fragments.music_fragment import MusicFragment


class Riff(MusicFragment):
    def __init__(self, path,):
        MusicFragment.__init__(self, path)


class GuitarRiff(Riff):
    def __init__(self, path, base_type='power_chord'):
        Riff.__init__(self, path)
        self.base_type = base_type


class BassRiff(Riff):
    def __init__(self, path, base_type='power_chord'):
        Riff.__init__(self, path)
        self.base_type = base_type
