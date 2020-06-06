from pymongo import MongoClient


def get_performers_table():
    client = MongoClient()
    return client.riff_db.performers


def get_albums_table():
    client = MongoClient()
    return client.riff_db.albums


def get_songs_table():
    client = MongoClient()
    return client.riff_db.songs


def get_guitar_riff_table():
    client = MongoClient()
    return client.riff_db.guitar_riff


def get_bass_riff_table():
    client = MongoClient()
    return client.riff_db.bass_riff


def get_unit_guitar_riff_table():
    client = MongoClient()
    return client.riff_db.unit_guitar_riff


def get_unit_bass_riff_table():
    client = MongoClient()
    return client.riff_db.unit_bass_riff


def get_guitar_solo_table():
    client = MongoClient()
    return client.riff_db.guitar_solo


def get_bass_solo_table():
    client = MongoClient()
    return client.riff_db.bass_solo


def find_empty():
    for table in [get_guitar_riff_table(), get_bass_riff_table(), get_guitar_solo_table()]:
        for piece in table.find({'Length': 0.0}):
            print(piece['Path'])


def find_long_riff():
    for table in [get_guitar_riff_table()]:
        for piece in table.find({'Length': {'$gt': 4}}):
            print(piece['Path'])


if __name__ == '__main__':
    find_long_riff()