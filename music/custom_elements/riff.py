from music.custom_elements.toolkit import *
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
        24 Acoustic Guitar (nylon)
        25 Acoustic Guitar (steel)
        26 Electric Guitar (jazz)
        27 Electric Guitar (clean)
        28 Electric Guitar (muted)
        29 Overdriven Guitar
        30 Distortion Guitar
        31 Guitar harmonics
        '''


class BassRiff(Riff):
    def __init__(self, measure_length, degrees_and_types, time_stamps, velocity=100):
        Riff.__init__(self, measure_length, degrees_and_types, time_stamps, velocity)

        '''
        32 Acoustic Bass
        33 Electric Bass (finger)
        34 Electric Bass (pick)
        35 Fretless Bass
        36 Slap Bass 1
        37 Slap Bass 2
        38 Synth Bass 1
        39 Synth Bass 2
        '''


def generate_briff_from_griff(guitar_riff):
    assert isinstance(guitar_riff, GuitarRiff)
    new_degrees_and_types = [(degree_and_type[0], '') for degree_and_type in guitar_riff.degrees_and_types]
    return BassRiff(guitar_riff.measure_length, new_degrees_and_types, guitar_riff.time_stamps, guitar_riff.velocity)

