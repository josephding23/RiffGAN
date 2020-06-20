from dataset.grunge_library import *
from music.db_fragments.riff import *
import os


def crop_riffs():
    griff_table = get_guitar_riff_table()
    briff_table = get_bass_riff_table()

    unit_griff_table = get_unit_guitar_riff_table()
    unit_briff_table = get_unit_bass_riff_table()

    for griff in griff_table.find():
        measures_tonality = griff['MeasuresTonality']

        path = griff['Path']
        guitar_riff = GuitarRiff(path)
        cropped_riffs = guitar_riff.crop_by_measure()

        if not os.path.exists(path[:-4]):
            os.mkdir(path[:-4])
        for measure in range(guitar_riff.measures_num):
            cropped = cropped_riffs[measure]
            save_path = path[:-4] + '/' + str(measure) + '.mid'

            cropped.write(path[:-4] + '/' + str(measure) + '.mid')

            unit_griff_table.insert_one({
                'Performer': griff['Performer'],
                'Album': griff['Album'],
                'Song': griff['Song'],
                'Path': save_path,
                'Tonality': measures_tonality[measure]
            })

    for briff in briff_table.find():
        measures_tonality = briff['MeasuresTonality']

        path = briff['Path']
        bass_riff = BassRiff(path)
        cropped_riffs = bass_riff.crop_by_measure()

        if not os.path.exists(path[:-4]):
            os.mkdir(path[:-4])
        for measure in range(bass_riff.measures_num):
            cropped = cropped_riffs[measure]
            save_path = path[:-4] + '/' + str(measure) + '.mid'

            cropped.write(path[:-4] + '/' + str(measure) + '.mid')

            unit_briff_table.insert_one({
                'Performer': briff['Performer'],
                'Album': briff['Album'],
                'Song': briff['Song'],
                'Path': save_path,
                'Tonality': measures_tonality[measure]
            })


def test_crop():
    path = 'E:/grunge_library/Soundgarden/Superunknown/03 - Fell on Black Days/RIFF/4.mid'
    guitar_riff = GuitarRiff(path)
    cropped_riffs = guitar_riff.crop_by_measure()
    os.mkdir(path[:-4])

    for measure in range(guitar_riff.measures_num):
        cropped = cropped_riffs[measure]
        cropped.write(path[:-4] + '/' + str(measure) + '.mid')


if __name__ == '__main__':
    crop_riffs()