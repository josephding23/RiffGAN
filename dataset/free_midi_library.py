from pymongo import MongoClient


def get_midi_table():
    client = MongoClient()
    return client.free_midi.midi

