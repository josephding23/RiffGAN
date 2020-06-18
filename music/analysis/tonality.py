from dataset.grunge_librarypy import *
from music.db_fragments.riff import *
from music.db_fragments.solo import *


def analyse_tonality():
    griff_table = get_guitar_riff_table()
    briff_table = get_bass_riff_table()
    gsolo_table = get_guitar_solo_table()

    for griff in griff_table.find():
        try:
            guitar_riff = GuitarRiff(griff['Path'])

            measures_num = guitar_riff.measures_num
            measures_tonality = guitar_riff.tonality_by_measures()

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
            measures_tonality = bass_riff.tonality_by_measures()

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
            measures_tonality = guitar_solo.tonality_by_measures()

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
    path = 'E:/database/test.mid'
    guitar_riff = GuitarRiff(path)
    print(guitar_riff.get_note_lengths_divided_by_measure())


if __name__ == '__main__':
    analyse_tonality()