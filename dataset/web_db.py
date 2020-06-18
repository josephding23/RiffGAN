from pymongo import MongoClient


def get_song_table():
    client = MongoClient()
    return client.riff_web.song
