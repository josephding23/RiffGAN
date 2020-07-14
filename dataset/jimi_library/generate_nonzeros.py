import numpy as np
from dataset.jimi_library.jimi_db import *
from music.db_fragments.riff import UnitBassRiff, UnitGuitarRiff


def generate_unit_riff_nonzeros():
    unit_guitar_riff_dir = 'E:/jimi_library/unit_riffs/transposed/guitar/'
    unit_bass_riff_dir = 'E:/jimi_library/unit_riffs/transposed/bass/'

    unit_guitar_nonzeros_dir = 'E:/jimi_library/unit_riffs/nonzeros/guitar/'
    unit_bass_nonzeros_dir = 'E:/jimi_library/unit_riffs/nonzeros/bass/'

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
        unit_bass_riff = UnitBassRiff(riff_path)
        unit_bass_riff.save_nonzeros(nonzeros_path)


def create_nonzeros():

    unit_guitar_nonzeros_dir = 'E:/jimi_library/unit_riffs/nonzeros/guitar/'
    unit_bass_nonzeros_dir = 'E:/jimi_library/unit_riffs/nonzeros/bass/'

    guitar_nonzeros_path = 'E:/jimi_library/data/guitar_unit_riff.npz'
    bass_nonzeros_path = 'E:/jimi_library/data/bass_unit_riff.npz'

    unit_guitar_table = get_unit_guitar_riff_table()
    unit_bass_table = get_unit_bass_riff_table()

    ugriff_num = unit_guitar_table.count()
    ubriff_num = unit_bass_table.count()

    unit_guitar_nonzeros = []
    unit_bass_nonzeros = []

    ugriff_no = 0
    for ugriff in unit_guitar_table.find():
        nonzeros_path = unit_guitar_nonzeros_dir + ugriff['idStr'] + '.npz'
        unit_nonzeros = np.load(nonzeros_path)['nonzeros']
        for nonzero in unit_nonzeros:
            unit_guitar_nonzeros.append([ugriff_no, nonzero[1], nonzero[2]])
            if nonzero[2] >= 60:
                print(nonzero)
        ugriff_no += 1

    ubriff_no = 0
    for ubriff in unit_bass_table.find():
        nonzeros_path = unit_bass_nonzeros_dir + ubriff['idStr'] + '.npz'
        nonzeros = np.load(nonzeros_path)['nonzeros']
        for nonzero in nonzeros:
            unit_bass_nonzeros.append([ubriff_no, nonzero[1], nonzero[2]])
        ubriff_no += 1

    np.savez_compressed(guitar_nonzeros_path, nonzeros=unit_guitar_nonzeros, shape=(ugriff_num, 64, 60))
    np.savez_compressed(bass_nonzeros_path, nonzeros=unit_bass_nonzeros, shape=(ubriff_num, 64, 48))


if __name__ == '__main__':
    generate_unit_riff_nonzeros()
    create_nonzeros()

