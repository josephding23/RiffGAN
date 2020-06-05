from src.fragments.music_fragment import MusicFragment


class Solo(MusicFragment):
    def __init__(self, path, source):
        MusicFragment.__init__(self, path, source)


class GuitarSolo(Solo):
    def __init__(self, path, source, base_type):
        Solo.__init__(self, path, source)
        self.base_type = base_type


class BassSolo(Solo):
    def __init__(self, path, source, base_type):
        Solo.__init__(self, path, source)
        self.base_type = base_type

