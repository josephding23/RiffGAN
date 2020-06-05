from util.database import *
from src.fragments.riff import *
from src.fragments.solo import *


def analyse_tonality():
    griff_table = get_guitar_riff_table()
    briff_table = get_bass_riff_table()
    gsolo_table = get_guitar_solo_table()

    for griff in griff_table.find():
        try:
            guitar_riff = GuitarRiff(griff['Path'])

            measures_num = guitar_riff.measures_num
            measures_tonality = guitar_riff.measures_tonality

            griff_table.update_one(
                {'_id': griff['_id']},
                {'$set': {
                    'MeasuresNum': measures_num,
                    'MeasuresTonality': measures_tonality
                }}
            )

        except:
            print(griff['Path'])

    for briff in briff_table.find():
        try:
            bass_riff = BassRiff(briff['Path'])

            measures_num = bass_riff.measures_num
            measures_tonality = bass_riff.measures_tonality

            briff_table.update_one(
                {'_id': briff['_id']},
                {'$set': {
                    'MeasuresNum': measures_num,
                    'MeasuresTonality': measures_tonality
                }}
            )

        except:
            print(briff['Path'])

    for gsolo in gsolo_table.find():
        try:
            guitar_solo = GuitarSolo(gsolo['Path'])

            measures_num = guitar_solo.measures_num
            measures_tonality = guitar_solo.measures_tonality

            gsolo_table.update_one(
                {'_id': gsolo['_id']},
                {'$set': {
                    'MeasuresNum': measures_num,
                    'MeasuresTonality': measures_tonality
                }}
            )

        except:
            print(gsolo['Path'])


def test_tonal():
    path = 'E:/data/test.mid'
    guitar_riff = GuitarRiff(path)
    print(guitar_riff.note_lengths_by_measures)


if __name__ == '__main__':
    analyse_tonality()