import mido
from dataset.grunge_library import *
from dataset.free_midi_library import *
import pretty_midi


def msg_sort_key(msg):
    return msg[1]


def time_info_sort(msg):
    return msg['time']


def get_meta_from_file(path):
    pm = pretty_midi.PrettyMIDI(path)
    meta_info = {
        'time_signature': [],
        'key_signature': [],
        'tempo_changes': []
    }

    for time_signature in pm.time_signature_changes:
        meta_info['time_signature'].append({
            'time': time_signature.time,
            'numerator': time_signature.numerator,
            'denominator': time_signature.denominator
        })

    for key_signature in pm.key_signature_changes:
        meta_info['key_signature'].append({
            'key_number': key_signature.key_number,
            'time': key_signature.time
        })

    tempo_changes_time, tempo_changes = pm.get_tempo_changes()[0], pm.get_tempo_changes()[1]
    for i in range(len(tempo_changes_time)):
        time = tempo_changes_time[i]
        tempo = tempo_changes[i]
        meta_info['tempo_changes'].append({
            'time': time,
            'tempo': tempo
        })

    return meta_info


def detect_grunge():
    song_table = get_songs_table()
    for song in song_table.find({'FileName': {'$exists': True}}):
        path = 'E:/grunge_library/midi_files/' + song['FileName']
        get_meta_from_file(path)
        print()


def detect_meta_in_free_midi():
    midi_table = get_midi_table()
    for midi in midi_table.find({'meta_info': {'$exists': False}}, no_cursor_timeout=True):
        path = 'E:/free_midi_library/raw_midi/' + midi['Genre'] + '/' + midi['md5'] + '.mid'
        meta_info = get_meta_from_file(path)
        midi_table.update_one(
            {'_id': midi['_id']},
            {'$set': {'meta_info': meta_info}}
        )
        print('Progress: {:.2%}\n'.format(midi_table.count({'meta_info': {'$exists': True}}) / midi_table.count()))


def set_bpm_in_free_midi():
    midi_table = get_midi_table()
    for midi in midi_table.find():
        meta_info = midi['meta_info']
        for i, bpm_info in enumerate(meta_info['tempo_changes']):
            bpm_info['round_tempo'] = round(bpm_info['tempo'], 2)
            meta_info['tempo_changes'][i] = bpm_info

        midi_table.update_one(
            {'_id': midi['_id']},
            {'$set': {'meta_info': meta_info}}
        )


def drop_duplicate_bpm():
    midi_table = get_midi_table()
    for midi in midi_table.find({'bpm_list': {'$exists': False}}):
        meta_info = midi['meta_info']
        bpm_list = meta_info['tempo_changes']

        clean_bpm_list = []

        for i, bpm_info in enumerate(bpm_list):
            bpm_info = {
                'time': bpm_info['time'],
                'tempo': bpm_info['round_tempo']
            }

            if bpm_info not in clean_bpm_list:
                clean_bpm_list.append(bpm_info)

        midi_table.update_one(
            {'_id': midi['_id']},
            {'$set': {'bpm_list': clean_bpm_list}}
        )

        print('Progress: {:.2%}\n'.format(midi_table.count({'bpm_list': {'$exists': True}}) / midi_table.count()))


def create_time_info_list():
    midi_table = get_midi_table()
    for midi in midi_table.find({'TimeInfoList': {'$exists': False}}):
        path = 'E:/free_midi_library/raw_midi/' + midi['Genre'] + '/' + midi['md5'] + '.mid'

        bpm_list = midi['BpmList']
        metre_list = midi['MetaInfo']['time_signature']

        whole_list = bpm_list + metre_list
        whole_list.sort(key=time_info_sort)

        info_list = []

        for i in range(len(whole_list)):
            for j in range(len(whole_list)):
                try:
                    if whole_list[i]['time'] == whole_list[j]['time']:
                        whole_list[i].update(whole_list[j])
                except:
                    pass

        for info in whole_list:
            if info not in info_list:
                info_list.append(info)

        midi_table.update_one(
            {'_id': midi['_id']},
            {'$set': {'TimeInfoList': info_list}}
        )


def rearrange_letter():
    midi_table = get_midi_table()
    for midi in midi_table.find():
        midi_table.update_one(
            {'_id': midi['_id']},
            {'$set': {'MetaInfo': midi['meta_info'], 'BpmList': midi['bpm_list']}}
        )

        midi_table.update_one(
            {'_id': midi['_id']},
            {'$unset': {'meta_info': '', 'bpm_list': ''}}
        )


def count_bar_length(bpm, metre):
    numerator, denominator = metre
    quarter_note_length = 60 / bpm
    denom_note_length = quarter_note_length * 4 / int(denominator)

    bar_length = denom_note_length * numerator
    return bar_length


if __name__ == '__main__':
    create_time_info_list()

