from music.pieces.phrase.drum_phrase import parse_drum_phrase_json
from music.pieces.phrase.rhythm_phrase import parse_rhythm_phrase_json
from music.pieces.phrase.toolkit import *


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


def refresh_phrase_info(track_info, phrase_dict):

    phrase_list = []

    for phrase_no in track_info['phrases_no']:
        if track_info['is_drum']:

            phrase = get_phrase_of_no(phrase_dict, 'drum_phrase', phrase_no)
            phrase_list.append(phrase)

        else:
            if track_info['instr_type'] == 'guitar':
                phrase = get_phrase_of_no(phrase_dict, 'rhythm_guitar_phrase', phrase_no)
                phrase_list.append(phrase)

            else:
                assert track_info['instr_type'] == 'bass'
                phrase = get_phrase_of_no(phrase_dict, 'rhythm_bass_phrase', phrase_no)
                phrase_list.append(phrase)

    track_info['phrases'] = phrase_list


def refresh_all_tracks(tracks, phrases):
    for track in tracks:
        refresh_phrase_info(track, phrases)