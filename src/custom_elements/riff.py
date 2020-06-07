import pretty_midi
from src.custom_elements.toolkit import *


class GuitarRiff:
    def __init__(self, length, root_note_name, degrees_and_types, time_stamps, velocity=100, instr=27):
        self.length = length

        self.root_note_name = root_note_name
        self.root_note = note_name_to_num(self.root_note_name)

        self.degrees_and_types = degrees_and_types
        self.time_stamps = time_stamps
        self.chords = [get_chord(degree_and_type) for degree_and_type in self.degrees_and_types]
        self.velocity = velocity

        self.instr = instr

        self.pm = pretty_midi.PrettyMIDI()
        self.add_notes_to_pm()

    def add_notes_to_pm(self):
        guitar = pretty_midi.Instrument(program=self.instr)
        for i in range(len(self.time_stamps)):
            start_time, end_time = self.time_stamps[i]
            if type(self.velocity) == int:
                velocity = self.velocity
            else:
                assert type(self.velocity) == list
                velocity = self.velocity[i]
            chord = self.chords[i]

            for note_dist in chord:
                note = pretty_midi.Note(velocity=velocity, pitch=note_dist+self.root_note,
                                        start=start_time, end=end_time)
                guitar.notes.append(note)

        self.pm.instruments.append(guitar)

    def save(self, save_path):
        self.pm.write(save_path)


def test_riff():
    griff = GuitarRiff(length=2.0, root_note_name='C3',
                       degrees_and_types=[('I', '5'), ('II', '5')],
                       time_stamps=[(0.0, 0.5), (0.5, 1.5)])
    griff.save('../../data/custom_element/guitar_riff/test1.mid')


if __name__ == '__main__':
    test_riff()