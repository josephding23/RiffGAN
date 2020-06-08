from music.db_fragments.music_fragment import MusicFragment


class Solo(MusicFragment):
    def __init__(self, path):
        MusicFragment.__init__(self, path)


class GuitarSolo(Solo):
    def __init__(self, path, base_type='power_chord'):
        Solo.__init__(self, path)
        self.base_type = base_type


class BassSolo(Solo):
    def __init__(self, path, base_type='power_chord'):
        Solo.__init__(self, path)
        self.base_type = base_type

