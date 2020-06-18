from music.custom_elements.riff.riff import Riff
from music.custom_elements.riff.guitar_riff import GuitarRiff
import json


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
