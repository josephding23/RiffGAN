import mido
from dataset.free_midi_library import *


def scan_rock_midi_files():
    midi_table = get_midi_table()
    genre = 'rock'
    dir = 'E:/free_midi_library/raw_midi/'
    for midi in midi_table.find({'Genre': genre}):
        path = dir + genre + '/' + midi['md5'] + '.mid'
        mido_object = mido.MidiFile(path)

        for i, track in enumerate(mido_object.tracks):
            for msg in track:
                if msg.is_meta and msg.type == 'set_tempo':
                    print(mido.tempo2bpm(msg.tempo))
        print('\n')


if __name__ == '__main__':
    scan_rock_midi_files()
