from src.fragments.music_fragment import MusicFragment


class Riff(MusicFragment):
    def __init__(self, path, source, tonic):
        MusicFragment.__init__(self, path, source)
        self.tonic = tonic


class GuitarRiff(Riff):
    def __init__(self, path, source, tonic='C', base_type='power_riff'):
        Riff.__init__(self, path, source, tonic)
        self.base_type = base_type


class BassRiff(Riff):
    def __init__(self, path, source, tonic='C', base_type='power_riff'):
        Riff.__init__(self, path, source, tonic)
        self.base_type = base_type
