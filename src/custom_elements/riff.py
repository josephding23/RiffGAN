import pretty_midi
from src.custom_elements.toolkit import *
from util.npy_related import *

class Riff:
    def __init__(self, measure_length, degrees_and_types, time_stamps, velocity):
        self.measure_length = measure_length

        self.degrees_and_types = degrees_and_types
        self.time_stamps = time_stamps
        self.chords = [get_chord(degree_and_type) for degree_and_type in self.degrees_and_types]
        self.velocity = velocity

        self.pm = pretty_midi.PrettyMIDI()

    def add_notes_to_pm(self, root_note_name, bpm, instr):
        root_note = note_name_to_num(root_note_name)
        guitar = pretty_midi.Instrument(program=instr)
        real_time_stamps = time_stamps_convert(self.time_stamps, bpm)
        for i in range(len(real_time_stamps)):
            start_time, end_time = real_time_stamps[i]
            if type(self.velocity) == int:
                velocity = self.velocity
            else:
                assert type(self.velocity) == list
                velocity = self.velocity[i]
            chord = self.chords[i]

            for note_dist in chord:
                note = pretty_midi.Note(velocity=velocity, pitch=note_dist+root_note,
                                        start=start_time, end=end_time)
                guitar.notes.append(note)

        self.pm.instruments.append(guitar)

    def save(self, save_path):
        self.pm.write(save_path)


class GuitarRiff(Riff):
    def __init__(self, measure_length, degrees_and_types, time_stamps, velocity=100):
        Riff.__init__(self, measure_length, degrees_and_types, time_stamps, velocity)

        '''
        25 Acoustic Guitar (nylon)
        26 Acoustic Guitar (steel)
        27 Electric Guitar (jazz)
        28 Electric Guitar (clean)
        29 Electric Guitar (muted)
        30 Overdriven Guitar
        31 Distortion Guitar
        32 Guitar harmonics
        '''


class BassRiff(Riff):
    def __init__(self, measure_length, degrees_and_types, time_stamps, velocity=100):
        Riff.__init__(self, measure_length, degrees_and_types, time_stamps, velocity)

        '''
        33 Acoustic Bass
        34 Electric Bass (finger)
        35 Electric Bass (pick)
        36 Fretless Bass
        37 Slap Bass 1
        38 Slap Bass 2
        39 Synth Bass 1
        40 Synth Bass 2
        '''


def test_riff():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1/2, 1/2, 1/2])
    griff.add_notes_to_pm(root_note_name='C3', bpm=120, instr=27)
    griff.save('../../data/custom_element/guitar_riff/test1.mid')


def test_plot():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    griff.add_notes_to_pm(root_note_name='C3', bpm=120, instr=27)

    nonzeros, shape = generate_nonzeros_from_pm(griff.pm, 120, 2)
    data = generate_sparse_matrix_from_nonzeros(nonzeros, shape)
    plot_data(data[0])


if __name__ == '__main__':
    test_plot()