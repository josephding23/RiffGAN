from util.npy_related import *
from music.db_fragments.riff import UnitGuitarRiff
from util.database import *


def generate_unit_riff_nonzeros():
    unit_guitar_riff_dir = 'E:/unit_riffs/transposed/guitar/'
    unit_bass_riff_dir = 'E:/unit_riffs/transposed/bass/'

    unit_guitar_nonzeros_dir = 'E:/unit_riffs/nonzeros/guitar/'
    unit_bass_nonzeros_dir = 'E:/unit_riffs/nonzeros/bass/'

    unit_guitar_table = get_unit_guitar_riff_table()
    unit_bass_table = get_unit_bass_riff_table()

    for ugriff in unit_guitar_table.find():
        riff_path = unit_guitar_riff_dir + ugriff['idStr'] + '.mid'
        nonzeros_path = unit_guitar_nonzeros_dir + ugriff['idStr'] + '.npz'
        unit_guitar_riff = UnitGuitarRiff(riff_path)
        unit_guitar_riff.save_nonzeros(nonzeros_path)

    for ubriff in unit_bass_table.find():
        riff_path = unit_bass_riff_dir + ubriff['idStr'] + '.mid'
        nonzeros_path = unit_bass_nonzeros_dir + ubriff['idStr'] + '.npz'
        unit_bass_riff = UnitGuitarRiff(riff_path)
        unit_bass_riff.save_nonzeros(nonzeros_path)


if __name__ == '__main__':
    generate_unit_riff_nonzeros()