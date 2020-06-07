from src.custom_elements.riff import *
from src.pieces.toolkit import *
import pretty_midi


class Phrase:
    def __init__(self, start_measure, length, tonality, bpm, instr):
        self.start_measure = start_measure
        self.length = length

        self.tonic, self.mode = tonality
        self.root_note = note_name_to_num(self.tonic)

        self.bpm = bpm
        self.instr = instr

        self.riffs = []
        self.arrangement = []

        self.pm = pretty_midi.PrettyMIDI()

    def add_riffs(self, riffs):
        self.riffs += riffs

    def set_arrangement(self, arrangement):
        self.arrangement = arrangement

    def add_riffs_to_pm(self):

        instr = pretty_midi.Instrument(program=self.instr)
        riff_start = 0
        length_per_measure = get_measure_length(self.bpm)
        for arrange in self.arrangement:
            riff = self.riffs[arrange]

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
                    note = pretty_midi.Note(velocity=velocity, pitch=note_dist + self.root_note,
                                            start=start_time, end=end_time)
                    instr.notes.append(note)

            riff_start += length_per_measure * riff.measure_length
            print(riff_start)

        self.pm.instruments.append(instr)

    def save(self, save_path):
        self.pm.write(save_path)


def test_phrase():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    phrase = Phrase(0, 6, ('C3', 'major'), 120, 26)
    phrase.add_riffs([griff])
    phrase.set_arrangement([0, 0, 0])
    phrase.add_riffs_to_pm()
    phrase.save('../../data/custom_element/phrase/test1.mid')


if __name__ == '__main__':
    test_phrase()