from music.custom_elements.riff import parse_griff_json, parse_briff_json
from music.custom_elements.drum_riff import parse_driff_json
from music.pieces.phrase import parse_rhythm_phrase_json, parse_drum_phrase_json
from music.custom_elements.toolkit import *


def get_measure_length(bpm):
    return 60 / bpm * 4


def set_used_riff_num_info(phrases_dict, riffs_dict):
    for i in range(len(phrases_dict['rhythm_guitar_phrase'])):
        used_no = []
        phrase_info = phrases_dict['rhythm_guitar_phrase'][i]
        for used_riff in phrase_info['riffs']:
            for reference_riff in riffs_dict['griff']:
                if parse_griff_json(used_riff) == parse_griff_json(reference_riff):
                    used_no.append(reference_riff['no'])
        phrase_info['riffs_no'] = used_no
        phrase_info['raw_riffs_no'] = ' '.join([str(no) for no in used_no])
        phrases_dict['rhythm_guitar_phrase'][i] = phrase_info

    for i in range(len(phrases_dict['rhythm_bass_phrase'])):
        used_no = []
        phrase_info = phrases_dict['rhythm_bass_phrase'][i]
        for used_riff in phrase_info['riffs']:
            for reference_riff in riffs_dict['briff']:
                if parse_briff_json(used_riff) == parse_briff_json(reference_riff):
                    used_no.append(reference_riff['no'])
        phrase_info['riffs_no'] = used_no
        phrase_info['raw_riffs_no'] = ' '.join([str(no) for no in used_no])
        phrases_dict['rhythm_bass_phrase'][i] = phrase_info

    for i in range(len(phrases_dict['drum_phrase'])):
        used_no = []
        phrase_info = phrases_dict['drum_phrase'][i]
        for used_riff in phrase_info['riffs']:
            for reference_riff in riffs_dict['driff']:
                if parse_driff_json(used_riff) == parse_driff_json(reference_riff):
                    used_no.append(reference_riff['no'])
        phrase_info['riffs_no'] = used_no
        phrase_info['raw_riffs_no'] = ' '.join([str(no) for no in used_no])
        phrases_dict['drum_phrase'][i] = phrase_info


def set_used_phrase_num_info(tracks_dict, phrases_dict):
    for i in range(len(tracks_dict)):
        used_no = []
        track_info = tracks_dict[i]

        if track_info['is_drum']:
            for used_phrase in track_info['phrases']:
                for reference_phrase in phrases_dict['drum_phrase']:
                    if parse_drum_phrase_json(used_phrase) == parse_drum_phrase_json(reference_phrase):
                        used_no.append(reference_phrase['no'])
            track_info['phrases_no'] = used_no
            track_info['raw_phrases_no'] = ' '.join([str(no) for no in used_no])
            tracks_dict[i] = track_info

        else:
            if track_info['instr_type'] == 'guitar':
                for used_phrase in track_info['phrases']:
                    for reference_phrase in phrases_dict['rhythm_guitar_phrase']:
                        if parse_rhythm_phrase_json(used_phrase) == parse_rhythm_phrase_json(reference_phrase):
                            used_no.append(reference_phrase['no'])
                track_info['phrases_no'] = used_no
                track_info['raw_phrases_no'] = ' '.join([str(no) for no in used_no])
                tracks_dict[i] = track_info

            else:
                assert track_info['instr_type'] == 'bass'
                for used_phrase in track_info['phrases']:
                    for reference_phrase in phrases_dict['rhythm_bass_phrase']:
                        if parse_rhythm_phrase_json(used_phrase) == parse_rhythm_phrase_json(reference_phrase):
                            used_no.append(reference_phrase['no'])
                track_info['phrases_no'] = used_no
                track_info['raw_phrases_no'] = ' '.join([str(no) for no in used_no])
                tracks_dict[i] = track_info


def get_used_riffs_from_raw(raw_used_riffs):
    used_riffs = []
    for used_riff in raw_used_riffs.split(' '):
        used_riffs.append(int(used_riff))
    return used_riffs


def get_rhythm_arrangements_from_raw(raw_arrangements):
    arrangements = []

    for arrangement in raw_arrangements.split('; '):
        start_time = int(arrangement.split(' ')[0])
        degree = arrangement.split(' ')[1]
        if get_relative_distance(degree) is None:
            raise Exception()
        else:
            arrangements.append([start_time, degree])
    return arrangements


def get_drum_arrangements_from_raw(raw_arrangements):
    arrangements = []
    for arrangement in raw_arrangements.split(' '):
        arrangements.append(int(arrangement))
    return arrangements


def get_available_riff_no(riff_dict, riff_type):
    available_no_list = []
    for riff_info in riff_dict[riff_type]:
        available_no_list.append(riff_info['no'])
    return available_no_list


def get_available_phrase_no(phrase_dict, phrase_type):
    available_no_list = []
    for phrase_info in phrase_dict[phrase_type]:
        available_no_list.append(phrase_info['no'])
    return available_no_list


def get_bpm_info_from_raw(raw_bpm_info):
    bpm_info_list = []
    for bpm_info in raw_bpm_info.split('; '):
        start_measure = int(bpm_info.split(' ')[0])
        bpm = int(bpm_info.split(' ')[1])

        bpm_info_list.append([start_measure, bpm])
    return bpm_info_list


def get_tonality_info_from_raw(raw_tonality_info):
    tonality_info_list = []
    for tonality_info in raw_tonality_info.split('; '):
        start_measure = int(tonality_info[0])
        tonality = tonality_info[1]

        tonality_info_list.append([start_measure, tonality])
    return tonality_info_list


def get_used_phrases_from_raw(raw_used_phrases):
    used_phrases = []
    for used_phrase in raw_used_phrases.split(' '):
        used_phrases.append(int(used_phrase))
    return used_phrases


def get_phrase_arrangements_from_raw(raw_arrangements):
    arrangements = []
    for arrangement in raw_arrangements.split('; '):
        phrase_no = int(arrangement.split(' ')[0])
        start_measure = int(arrangement.split(' ')[1])
        arrangements.append([phrase_no, start_measure])
    return arrangements