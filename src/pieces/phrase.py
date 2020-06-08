from src.custom_elements.riff import *
from src.custom_elements.drum_riff import *
from src.pieces.toolkit import *
import pretty_midi


class Phrase:
    def __init__(self, start_measure, length, bpm):
        self.start_measure = start_measure
        self.length = length

        self.bpm = bpm

        self.pm = None

    def save(self, save_path):
        assert self.pm is not None
        self.pm.write(save_path)


class DrumPhrase(Phrase):
    def __init__(self, start_measure, length, bpm):
        Phrase.__init__(self, start_measure, length, bpm)

        self.riffs = []
        self.arrangement = []

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
                if pattern is None:
                    continue
                else:
                    assert isinstance(pattern, str)

                    total_num = len(pattern)
                    measure_length = get_measure_length(self.bpm)
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


class RiffPhrase(Phrase):
    def __init__(self, start_measure, length, bpm, instr, tonality):
        Phrase.__init__(self, start_measure, length, bpm)

        self.tonic, self.mode = tonality
        self.root_note = note_name_to_num(self.tonic)

        self.instr = instr

        self.riffs = []
        self.arrangement = []

    def set_riffs(self, riffs):
        self.riffs = riffs

    def set_arrangement(self, arrangement):
        self.arrangement = arrangement

    def add_riffs_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()

        instr = pretty_midi.Instrument(program=self.instr)
        riff_start = 0
        length_per_measure = get_measure_length(self.bpm)

        for arrange in self.arrangement:
            riff, riff_root_name = self.riffs[arrange[0]], arrange[1]
            riff_root_dist = get_relative_distance(riff_root_name)

            real_time_stamps = time_stamps_convert(riff.time_stamps, self.bpm)
            for i in range(len(real_time_stamps)):
                start_time, end_time = real_time_stamps[i]
                start_time += riff_start
                end_time += riff_start
                if type(riff.velocity) == int:
                    velocity = riff.velocity
                else:
                    assert type(riff.velocity) == list
                    velocity = riff.velocity[i]
                chord = riff.chords[i]

                for note_dist in chord:
                    note = pretty_midi.Note(velocity=velocity, pitch=note_dist + self.root_note + riff_root_dist,
                                            start=start_time, end=end_time)
                    instr.notes.append(note)

            riff_start += length_per_measure * riff.measure_length
            # print(riff_start)

        self.pm.instruments.append(instr)


class SoloPhrase(Phrase):
    def __init__(self, start_measure, length, bpm, instr, tonality):
        Phrase.__init__(self, start_measure, length, bpm)

        self.tonic, self.mode = tonality
        self.root_note = note_name_to_num(self.tonic)

        self.instr = instr


def test_drum_phrase():
    drum_riff = DrumRiff(measure_length=1)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})

    drum_phrase = DrumPhrase(0, length=3, bpm=120)
    drum_phrase.set_riffs([drum_riff])
    drum_phrase.set_arrangement([0, 0, 0])
    drum_phrase.add_riffs_to_pm()
    drum_phrase.save('../../data/custom_element/phrase/test2.mid')


if __name__ == '__main__':
    test_drum_phrase()
