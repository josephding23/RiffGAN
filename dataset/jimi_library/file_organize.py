import os
from dataset.jimi_library.jimi_db import *
import shutil


def reorganize_grunge_midi():
    song_table = get_songs_table()
    root_dir = 'E:/jimi_library/raw'
    dst_dir = 'E:/jimi_library/midi_files'

    for song in song_table.find():
        dir = root_dir + '/' + song['Performer'] + '/' + song['Album'] + '/' + song['TrackNum'] + ' - ' + song['Name']
        for f in os.listdir(dir):
            if f[-4:] == '.mid':
                new_name = song['Performer'] + ' - ' + song['Album'] + ' - ' + song['TrackNum'] + ' - ' + song['Name'] + '.mid'
                song_table.update_one(
                    {'_id': song['_id']},
                    {'$set': {
                        'FileName': new_name
                    }}
                )
                os.chdir(dir)
                os.rename(f, new_name)
                move_file(dir, dst_dir, new_name)


def set_song_name():
    song_table = get_songs_table()
    for song in song_table.find():
        new_name = song['Performer'] + ' - ' + song['Album'] + ' - ' + song['TrackNum'] + ' - ' + song['Name'] + '.mid'
        song_table.update_one(
            {'_id': song['_id']},
            {'$set': {
                'FileName': new_name
            }}
        )


def move_file(src_path, dst_path, file):
    try:
        f_src = os.path.join(src_path, file)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        f_dst = os.path.join(dst_path, file)
        shutil.move(f_src, f_dst)
    except Exception as e:
        pass


if __name__ == '__main__':
    set_song_name()