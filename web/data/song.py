from music.pieces.phrase.toolkit import *
from music.pieces.track.toolkit import *
from music.pieces.song.song import create_song_drom_json

json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'

song = create_song_drom_json(json_path)
riffs = song.get_all_riffs()
phrases = song.get_all_phrases()
tracks = song.get_all_tracks()

loaded = False

set_used_riff_num_info(phrases, riffs)
set_used_phrase_num_info(tracks, phrases)
