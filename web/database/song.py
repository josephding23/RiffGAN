from music.pieces.song.song import *
from music.pieces.song.song import create_song_from_json, get_empty_song, parse_song_json
import json
from dataset.web_db import get_song_table

json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'


def load_song_and_duplicate_as_temp(name):
    song_info = load_song_from_db(name)

    song = parse_song_json(song_info)
    song.name = 'temp'

    song.save_to_db()


def create_new_song_and_duplicate_as_temp():
    song_info = get_empty_song()
    song = parse_song_json(song_info)

    song.name = 'temp'
    song.save_to_db()


def save_temp_song_as(name):
    song_info = get_temp_song()
    song = parse_song_json(song_info)

    song.name = name
    song.save_to_db()


def save_temp_song(song):
    get_song_table().update_one(
        {'name': 'temp'},
        {'$set': song}
    )


def get_temp_song():
    return load_song_from_db('temp')


def save_temp_riffs(riffs):
    print(get_temp_riffs())
    get_song_table().update_one(
        {'name': 'temp'},
        {'$set': {'riffs': riffs}}
    )
    print(get_temp_riffs())


def get_temp_riffs():
    song = get_song_table().find_one({'name': 'temp'})
    return song['riffs']


def save_temp_phrases(phrases):
    get_song_table().update_one(
        {'name': 'temp'},
        {'$set': {'phrases': phrases}}
    )


def get_temp_phrases():
    return load_song_from_db('temp')['phrases']


def save_temp_tracks(tracks):
    get_song_table().update_one(
        {'name': 'temp'},
        {'$set': {'tracks': tracks}}
    )


def get_temp_tracks():
    return load_song_from_db('temp')['tracks']