from music.custom_elements.rhythm_riff.riff import Riff
from dataset.web_db import get_griff_table
import json


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

    def save_to_db(self, name):
        riff_table = get_griff_table()
        riff_info = self.export_json_dict()

        riff_info['name'] = name

        if riff_table.find_one({'name': name}) is None:
            riff_table.insert_one(riff_info)
        else:
            riff_table.update_one(
                {'name': name},
                {'$set': riff_info}
            )


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
