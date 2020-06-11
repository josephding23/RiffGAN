from music.custom_elements.toolkit import *
from util.npy_related import *
import pretty_midi


class DrumRiff:
    def __init__(self, measure_length):
        self.measure_length = measure_length

        self.patterns = {
            'hi-hat': None,
            # closed_hi-hat: 42, pedal_hi-hat: 44, open_hi-hat: 46
            'snare': None,
            # acoustic_snare: 38
            'bass': None,
            # acoustic_bass: 35
            'tom': None,
            # low_floor_tom: 41, high_floor_tom: 43,
            # low_tom: 45, low-mid_tom: 47,
            # hi-mid_tom: 48, high_tom: 50
            'ride': None,
            # ride_cymbal1: 51, ride_cymbal2: 59
            'crash': None,
            # crash_cymbal1: 49, crash_cymbal2: 57
            'splash': None
            # splash_cymbal: 55
        }

        self.pm = None
        self.save_dir = '../../../data/custom_element/drum_riff/'

    def set_specific_pattern(self, part, pattern):
        self.patterns[part] = pattern

    def set_pattern(self, patterns):
        for part, pattern in patterns.items():
            self.patterns[part] = pattern

    def add_specific_pattern_to_pm(self, part, bpm):
        self.pm = pretty_midi.PrettyMIDI()
        pattern = self.patterns[part]
        assert isinstance(pattern, str)

        total_num = len(pattern)
        measure_length = get_measure_length(bpm) * self.measure_length
        unit_length = measure_length / total_num

        drum = pretty_midi.Instrument(program=0, is_drum=True)
        for i in range(total_num):
            symbol = pattern[i]
            if symbol == '_':
                continue
            else:
                start_time, end_time = i * unit_length, (i+1) * unit_length
                note = pretty_midi.Note(velocity=100, pitch=translate_symbol(part, symbol),
                                        start=start_time, end=end_time)
                drum.notes.append(note)

        self.pm.instruments.append(drum)

    def add_all_patterns_to_pm(self, bpm):
        self.pm = pretty_midi.PrettyMIDI()
        drum = pretty_midi.Instrument(program=0, is_drum=True)

        for part, pattern in self.patterns.items():
            if pattern is None:
                continue
            else:
                assert isinstance(pattern, str)

                total_num = len(pattern)
                measure_length = get_measure_length(bpm) * self.measure_length
                unit_length = measure_length / total_num

                for i in range(total_num):
                    symbol = pattern[i]
                    if symbol == '_':
                        continue
                    else:
                        start_time, end_time = i * unit_length, (i + 1) * unit_length
                        note = pretty_midi.Note(velocity=100, pitch=translate_symbol(part, symbol),
                                                start=start_time, end=end_time)
                        drum.notes.append(note)

        self.pm.instruments.append(drum)

    def save(self, name):
        self.pm.write(self.save_dir + name)


def translate_symbol(part, symbol):
    symbol_dict = {
        'hi-hat':
            {'c': 42,  'p': 44, 'o': 46},
        'snare':
            {'x': 38},
        'bass':
            {'x': 35},
        'tom':
            {'1': 41, '2': 43, '3': 45, '4': 47, '5': 48, '6': 50},
        'ride':
            {'1': 51, '2': 59},
        'crash':
            {'1': 49, '2': 57},
        'splash':
            {'x': 55}
    }

    return symbol_dict[part][symbol]
