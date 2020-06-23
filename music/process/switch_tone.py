from dataset.grunge_library import *
from music.db_fragments.riff import *
import shutil


def get_key_num(tonic):
    tonic_list = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭g', 'G', '♭A', 'A', '♭B', 'B']
    return tonic_list.index(tonic)


def get_near_distance(tonic1, tonic2):
    key1, key2 = get_key_num(tonic1), get_key_num(tonic2)
    semitones = key1 - key2
    if semitones > 0:
        alt_semitones = abs(key1 - key2) - 12
    else:
        alt_semitones = 12 - abs(key1 - key2)
    if abs(semitones) <= 6:
        pass
    else:
        semitones = alt_semitones

    return semitones


def add_hash():
    riff_tables = [get_unit_guitar_riff_table(), get_unit_bass_riff_table()]
    for riff_table in riff_tables:
        for riff in riff_table.find():
            riff_table.update_one(
                {'_id': riff['_id']},
                {'$set': {
                    'idStr': str(riff['_id'])
                }}
            )


def move_to_new_folder():
    griff_table, briff_table = get_unit_guitar_riff_table(), get_unit_bass_riff_table()

    griff_dir = 'E:/unit_riffs/original/guitar'
    briff_dir = 'E:/unit_riffs/original/bass'

    for griff in griff_table.find():
        idStr = griff['idStr']
        new_path = griff_dir + '/' + idStr + '.mid'
        shutil.copyfile(griff['Path'], new_path)

    for briff in briff_table.find():
        idStr = briff['idStr']
        new_path = briff_dir + '/' + idStr + '.mid'
        shutil.copyfile(briff['Path'], new_path)


def transpose_to_c():
    griff_table, briff_table = get_unit_guitar_riff_table(), get_unit_bass_riff_table()

    griff_ori_dir = 'E:/grunge_library/unit_riffs/original/guitar'
    briff_ori_dir = 'E:/grunge_library/unit_riffs/original/bass'

    griff_transposed_dir = 'E:/grunge_library/unit_riffs/transposed/guitar'
    briff_transposed_dir = 'E:/grunge_library/unit_riffs/transposed/bass'

    for griff in griff_table.find():
        idStr = griff['idStr']
        tonic = griff['Tonality']['Tonic']
        semitones = get_near_distance('C', tonic)

        old_path = griff_ori_dir + '/' + idStr + '.mid'
        new_path = griff_transposed_dir + '/' + idStr + '.mid'
        unit_guitar_riff = UnitGuitarRiff(old_path)

        pm = unit_guitar_riff.pm
        for instr in pm.instruments:
            if not instr.is_drum:
                for _note in instr.notes:
                    if _note.pitch + semitones in range(0, 128):
                        _note.pitch += semitones
        pm.write(new_path)

    for briff in briff_table.find():
        idStr = briff['idStr']
        tonic = briff['Tonality']['Tonic']
        semitones = get_near_distance('C', tonic)

        old_path = briff_ori_dir + '/' + idStr + '.mid'
        new_path = briff_transposed_dir + '/' + idStr + '.mid'
        unit_guitar_riff = UnitGuitarRiff(old_path)

        pm = unit_guitar_riff.pm
        for instr in pm.instruments:
            if not instr.is_drum:
                for _note in instr.notes:
                    if _note.pitch + semitones in range(0, 128):
                        _note.pitch += semitones
        pm.write(new_path)


if __name__ == '__main__':
    transpose_to_c()