from music.custom_elements.riff.guitar_riff import parse_griff_json
from music.custom_elements.riff.bass_riff import parse_briff_json
from music.custom_elements.riff.drum_riff import parse_driff_json, get_relative_distance
from music.custom_elements.riff.toolkit import get_riff_of_no


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


def refresh_riff_info(phrase_info, phrase_type, riff_dict):

    according_riff_dict = {
        'rhythm_guitar_phrase': 'griff',
        'rhythm_bass_phrase': 'briff',
        'drum_phrase': 'driff'
    }

    riff_list = []

    for riff_no in phrase_info['riffs_no']:
        riff = get_riff_of_no(riff_dict, according_riff_dict[phrase_type], riff_no)
        riff_list.append(riff)

    phrase_info['riffs'] = riff_list


def refresh_all_phrases(phrases, riffs):
    for phrase_type in ['rhythm_guitar_phrase', 'rhythm_bass_phrase', 'drum_phrase']:
        phrases_list = phrases[phrase_type]
        for phrase in phrases_list:
            refresh_riff_info(phrase, phrase_type, riffs)


def get_phrase_of_no(phrase_dict, phrase_type, no):
    for phrase_info in phrase_dict[phrase_type]:
        current_no = phrase_info['no']
        if current_no == no:
            return phrase_info
