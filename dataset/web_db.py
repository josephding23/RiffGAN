from pymongo import MongoClient


def get_song_table():
    client = MongoClient()
    return client.riff_web.song


def get_griff_table():
    client = MongoClient()
    return client.riff_web.griff


def get_briff_table():
    client = MongoClient()
    return client.riff_web.briff


def get_driff_table():
    client = MongoClient()
    return client.riff_web.driff


def get_rhythm_guitar_phrase_table():
    client = MongoClient()
    return client.riff_web.rhythm_guitar_phrase


def get_rhythm_bass_phrase_table():
    client = MongoClient()
    return client.riff_web.rhythm_bass_phrase


def get_drum_phrase_table():
    client = MongoClient()
    return client.riff_web.drum_phrase


def get_track_table():
    client = MongoClient()
    return client.riff_web.track
