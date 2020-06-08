from music.db_fragments.music_fragment import MusicFragment


class Riff(MusicFragment):
    def __init__(self, path):
        MusicFragment.__init__(self, path)


class GuitarRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)


class BassRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)


class UnitGuitarRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)


class UnitBassRiff(Riff):
    def __init__(self, path):
        Riff.__init__(self, path)
