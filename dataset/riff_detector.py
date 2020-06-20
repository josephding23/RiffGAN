import mido
from dataset.grunge_library import *
import pretty_midi


def get_tempo_from_file(path):
    mid = mido.MidiFile(path)
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.is_meta and msg.type == 'set_tempo':
                print(mido.tempo2bpm(msg.tempo), msg.time)


def detect_grunge():
    song_table = get_songs_table()
    for song in song_table.find({'FileName': {'$exists': True}}):
        path = 'E:/grunge_library/midi_files/' + song['FileName']
        get_tempo_from_file(path)
        print()


if __name__ == '__main__':
    detect_grunge()