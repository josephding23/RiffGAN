from src.fragments.riff import GuitarRiff, BassRiff
from src.fragments.solo import Solo
from pymongo import MongoClient
import os
import math
import traceback

def get_guitar_riff_table():
    client = MongoClient()
    return client.riff_db.guitar_riff


def get_bass_riff_table():
    client = MongoClient()
    return client.riff_db.bass_riff


def get_guitar_solo_table():
    client = MongoClient()
    return client.riff_db.guitar_solo


def get_bass_solo_table():
    client = MongoClient()
    return client.riff_db.bass_solo


def scan_local_files():
    root_dir = 'E:/grunge_library'
    for performer in os.listdir(root_dir):
        performer_dir = root_dir + '/' + performer

        for album in os.listdir(performer_dir):
            album_dir = performer_dir + '/' + album

            for song in os.listdir(album_dir):
                song_dir = album_dir + '/' + song

                guitar_riff_dir = song_dir + '/' + 'RIFF'
                bass_riff_dir = song_dir + '/' + 'BASS'
                guitar_solo_dir = song_dir + '/' + 'SOLO'

                if os.path.exists(guitar_riff_dir):
                    for griff in os.listdir(guitar_riff_dir):
                        griff_path = guitar_riff_dir + '/' + griff

                        try:
                            guitar_riff = GuitarRiff(griff_path, 'null')
                            riff_length = guitar_riff.end_time
                            if math.modf(riff_length)[0] != 0.0:
                                print(riff_length)
                                print(griff_path)
                        except:
                            print(griff_path)
                            print(traceback.format_exc())
                if os.path.exists(bass_riff_dir):
                    pass

                if os.path.exists(guitar_solo_dir):
                    pass



if __name__ == '__main__':
    scan_local_files()