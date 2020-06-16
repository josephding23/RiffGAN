from music.pieces.phrase.phrase import *
from music.pieces.phrase.toolkit import *
from music.custom_elements.riff.drum_riff import *


class DrumPhrase(Phrase):
    def __init__(self, length, bpm):
        Phrase.__init__(self, length, bpm)

        self.riffs = []
        self.arrangement = []

    def __eq__(self, other):
        assert isinstance(other, DrumPhrase)

        return self.length == other.length and self.bpm == other.bpm and \
               self.riffs == other.riffs and self.arrangement == other.arrangement

    def set_riffs(self, riffs):
        self.riffs = riffs

    def set_arrangement(self, arrangement):
        self.arrangement = arrangement

    def add_riffs_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()

        drum = pretty_midi.Instrument(program=0, is_drum=True)
        riff_start = 0
        length_per_measure = get_measure_length(self.bpm)

        for arrange in self.arrangement:
            riff = self.riffs[arrange]
            for part, pattern in riff.patterns.items():
                if pattern is '':
                    continue
                else:
                    assert isinstance(pattern, str)

                    total_num = len(pattern)
                    measure_length = get_measure_length(self.bpm) * riff.measure_length
                    unit_length = measure_length / total_num

                    for i in range(total_num):
                        symbol = pattern[i]
                        if symbol == '_':
                            continue
                        else:
                            start_time, end_time = i * unit_length, (i + 1) * unit_length
                            start_time += riff_start
                            end_time += riff_start

                            note = pretty_midi.Note(velocity=100, pitch=translate_symbol(part, symbol),
                                                    start=start_time, end=end_time)
                            drum.notes.append(note)

            riff_start += length_per_measure * riff.measure_length

        self.pm.instruments.append(drum)

    def get_arrangement_str(self):
        info_str = ''
        for arrangement in self.arrangement:
            info_str += str(arrangement) + ' '
        return info_str[:-1]

    def export_json_dict(self):
        info_dict = {
            "length": self.length,
            "bpm": self.bpm,
            "riffs": [riff.export_json_dict() for riff in self.riffs],
            "arrangements": self.arrangement
        }
        return info_dict


def create_drum_phrase_from_json(path):
    with open(path, 'r') as f:
        phrase_info = json.loads(f.read())
        return parse_drum_phrase_json(phrase_info)


def parse_drum_phrase_json(phrase_info):
    drum_phrase = DrumPhrase(
        length=phrase_info['length'],
        bpm=phrase_info['bpm'],
    )

    drum_phrase.set_riffs([parse_driff_json(riff_info) for riff_info in phrase_info['riffs']])
    drum_phrase.set_arrangement(phrase_info['arrangements'])

    return drum_phrase
