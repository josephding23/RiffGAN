import pretty_midi


class MusicFragment:
    def __init__(self, path, source):
        self.path = path
        self.source = source
        self.pm = pretty_midi.PrettyMIDI(path)
        self.end_time = self.pm.get_end_time()