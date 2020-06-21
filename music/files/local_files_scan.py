from music.db_fragments.riff import GuitarRiff, BassRiff
from music.db_fragments.solo import GuitarSolo
import os
import traceback
from dataset.grunge_library import *


def scan_local_files():
    root_dir = 'E:/grunge_library'

    performers_table = get_performers_table()
    albums_table = get_albums_table()
    songs_table = get_songs_table()
    griff_table = get_guitar_riff_table()
    briff_table = get_bass_riff_table()
    gsolo_table = get_guitar_solo_table()

    for performer in os.listdir(root_dir):
        performer_dir = root_dir + '/' + performer

        album_num = len(os.listdir(performer_dir))
        performer_song_num = 0
        performer_griff_num = 0
        performer_briff_num = 0
        performer_gsolo_num = 0

        for album in os.listdir(performer_dir):
            album_dir = performer_dir + '/' + album

            album_song_num = len(os.listdir(album_dir))
            album_griff_num = 0
            album_briff_num = 0
            album_gsolo_num = 0

            for song in os.listdir(album_dir):
                song_dir = album_dir + '/' + song

                guitar_riff_dir = song_dir + '/' + 'RIFF'
                bass_riff_dir = song_dir + '/' + 'BASS'
                guitar_solo_dir = song_dir + '/' + 'SOLO'

                # Read Guitar Riff
                if os.path.exists(guitar_riff_dir):
                    song_griff_num = len(os.listdir(guitar_riff_dir))
                    for griff in os.listdir(guitar_riff_dir):
                        griff_path = guitar_riff_dir + '/' + griff

                        try:
                            guitar_riff = GuitarRiff(griff_path, 'null')
                            riff_length = guitar_riff.length

                            griff_table.insert_one({
                                'Performer': performer,
                                'Album': album,
                                'Song': song,
                                'Path': griff_path,
                                'Length': riff_length
                            })

                        except:
                            print(griff_path)
                            print(traceback.format_exc())

                else:
                    song_griff_num = 0

                # Read Bass Riff
                if os.path.exists(bass_riff_dir):
                    song_briff_num = len(os.listdir(bass_riff_dir))
                    for briff in os.listdir(bass_riff_dir):
                        briff_path = bass_riff_dir + '/' + briff

                        try:
                            bass_riff = BassRiff(briff_path, 'null')
                            riff_length = bass_riff.length

                            briff_table.insert_one({
                                'Performer': performer,
                                'Album': album,
                                'Song': song,
                                'Path': briff_path,
                                'Length': riff_length
                            })


                        except:
                            print(briff_path)
                            print(traceback.format_exc())
                else:
                    song_briff_num = 0

                # Read Guitar Solo
                if os.path.exists(guitar_solo_dir):
                    song_gsolo_num = len(os.listdir(guitar_solo_dir))
                    for gsolo in os.listdir(guitar_solo_dir):
                        gsolo_path = guitar_solo_dir + '/' + gsolo

                        try:
                            guitar_solo = GuitarSolo(gsolo_path, 'null')
                            solo_length = guitar_solo.length

                            gsolo_table.insert_one({
                                'Performer': performer,
                                'Album': album,
                                'Song': song,
                                'Path': gsolo_path,
                                'Length': solo_length
                            })


                        except:
                            print(gsolo_path)
                            print(traceback.format_exc())
                else:
                    song_gsolo_num = 0

                songs_table.insert_one({
                    'Name': song.split(' - ')[1],
                    'TrackNum': song.split(' - ')[0],
                    'Album': album,
                    'Performer': performer,
                    'GuitarRiffNum': song_griff_num,
                    'BassRiffNum': song_briff_num,
                    'GuitarSoloNum': song_gsolo_num
                })

                album_griff_num += song_griff_num
                album_briff_num += song_briff_num
                album_gsolo_num += song_gsolo_num

            albums_table.insert_one({
                'Name': album,
                'TrackNum': album_song_num,
                'Songs': [song.split(' - ')[1] for song in os.listdir(album_dir)],
                'Performer': performer,
                'GuitarRiffNum': album_griff_num,
                'BassRiffNum': album_briff_num,
                'GuitarSoloNum': album_gsolo_num
            })

            performer_song_num += album_song_num
            performer_griff_num += album_griff_num
            performer_briff_num += album_briff_num
            performer_gsolo_num += album_gsolo_num

        performers_table.insert_one({
            'Name': performer,
            'Albums': len(os.listdir(performer_dir)),
            'AlbumNum': album_num,
            'GuitarRiffNum': performer_griff_num,
            'BassRiffNum': performer_briff_num,
            'GuitarSoloNum': performer_gsolo_num
        })


if __name__ == '__main__':
    scan_local_files()