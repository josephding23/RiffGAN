from music.custom_elements.toolkit import *
from util.npy_related import *
import json
import os
from music.process.audio_related import play_music


class Riff:
    def __init__(self, measure_length, degrees_and_types, time_stamps, velocity):
        self.measure_length = measure_length

        self.degrees_and_types = degrees_and_types

        self.save_dir = ''
        self.midi_path = ''
        self.saved = False

        self.time_stamps = time_stamps
        self.chords = [get_chord(degree_and_type) for degree_and_type in self.degrees_and_types]
        self.velocity = velocity

        self.pm = pretty_midi.PrettyMIDI()

    def __eq__(self, other):
        return self.measure_length == other.measure_length and self.degrees_and_types == other.degrees_and_types and \
               self.time_stamps == other.time_stamps and self.velocity == other.velocity

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
                note = pretty_midi.Note(velocity=velocity, pitch=note_dist + root_note,
                                        start=start_time, end=end_time)
                guitar.notes.append(note)

        self.pm.instruments.append(guitar)

    def get_degrees_and_types_str(self):
        info_str = ''
        for degree_and_type in self.degrees_and_types:
            info_str += degree_and_type[0] + ' ' + degree_and_type[1] + '; '
        return info_str[:-2]

    def get_timestamps_str(self):
        info_str = ''
        for timestamp in self.time_stamps:
            info_str += str(timestamp) + ' '
        return info_str[:-1]

    def save_midi(self, name):
        self.midi_path = self.save_dir + name + '.mid'
        self.pm.write(self.midi_path)

    def play_it(self):
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music(self.midi_path)

    def export_json_dict(self):
        info_dict = {
            "length": self.measure_length,
            "degrees_and_types": self.degrees_and_types,
            "time_stamps": self.time_stamps
        }
        return info_dict

    def save_json(self, name):
        assert self.save_dir != ''
        with open(self.save_dir + 'json/' + name + '.json', 'w') as f:
            json.dump(self.export_json_dict(), f)


class GuitarRiff(Riff):
    def __init__(self, measure_length, degrees_and_types, time_stamps, velocity=100):
        Riff.__init__(self, measure_length, degrees_and_types, time_stamps, velocity)
        self.save_dir = 'D:/PycharmProjects/RiffGAN/data/custom_element/guitar_riff/'
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
        self.save_dir = 'D:/PycharmProjects/RiffGAN/data/custom_element/bass_riff/'

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


def create_griff_from_json(path):
    with open(path, 'r') as f:
        riff_info = json.loads(f.read())
        return parse_griff_json(riff_info)


def parse_griff_json(riff_info):
    return GuitarRiff(
        measure_length=riff_info['length'],
        degrees_and_types=riff_info['degrees_and_types'],
        time_stamps=riff_info['time_stamps']
    )


def create_briff_from_json(path):
    with open(path, 'r') as f:
        riff_info = json.loads(f.read())
        return parse_briff_json(riff_info)


def parse_briff_json(riff_info):
    return BassRiff(
        measure_length=riff_info['length'],
        degrees_and_types=riff_info['degrees_and_types'],
        time_stamps=riff_info['time_stamps']
    )


def generate_briff_from_griff(guitar_riff):
    assert isinstance(guitar_riff, GuitarRiff)
    new_degrees_and_types = [(degree_and_type[0], '') for degree_and_type in guitar_riff.degrees_and_types]
    return BassRiff(guitar_riff.measure_length, new_degrees_and_types, guitar_riff.time_stamps, guitar_riff.velocity)
