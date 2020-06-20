from music.pieces.song.song import *
from music.pieces.song.song import parse_griff_json, parse_briff_json, parse_driff_json
import json
from dataset.web_db import get_griff_table, get_briff_table, get_driff_table

json_path = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/json/test_song.json'


def exists_riff(name, riff_type):
    if riff_type == 'griff':
        riff_table = get_griff_table()
    elif riff_type == 'briff':
        riff_table = get_briff_table()
    else:
        assert riff_type == 'driff'
        riff_table = get_driff_table()
    return riff_table.find_one({'name': name}) is not None


def save_riff_as(name, riff_info, riff_type):
    if riff_type == 'griff':
        riff = parse_griff_json(riff_info)
    elif riff_type == 'briff':
        riff = parse_briff_json(riff_info)
    else:
        assert riff_type == 'driff'
        riff = parse_driff_json(riff_info)
    riff.save_to_db(name)


def get_all_existed_riffs(riff_type):
    riff_list = []
    if riff_type == 'griff':
        riff_table = get_griff_table()
    elif riff_type == 'briff':
        riff_table = get_briff_table()
    else:
        assert riff_type == 'driff'
        riff_table = get_driff_table()

    for riff in riff_table.find():
        riff_list.append(riff)

    return riff_list


def delete_stored_riff_in_db(name, riff_type):
    if riff_type == 'griff':
        riff_table = get_griff_table()
    elif riff_type == 'briff':
        riff_table = get_briff_table()
    else:
        assert riff_type == 'driff'
        riff_table = get_driff_table()

    riff_table.remove({'name': name})