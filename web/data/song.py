from music.pieces.song import *
from music.custom_elements.toolkit import *
from music.pieces.toolkit import *

json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'

song = create_song_drom_json(json_path)
riffs = song.get_all_riffs()
phrases = song.get_all_phrases()
tracks = song.get_all_tracks()

set_used_riff_num_info(phrases, riffs)
