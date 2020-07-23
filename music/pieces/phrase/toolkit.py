from music.custom_elements.rhythm_riff.guitar_riff import parse_griff_json
from music.custom_elements.rhythm_riff.bass_riff import parse_briff_json
from music.custom_elements.drum_riff.drum_riff import parse_driff_json, get_relative_distance
from music.custom_elements.modified_riff.modified_riff import parse_modified_griff_json, parse_modified_briff_json
from music.custom_elements.rhythm_riff.toolkit import get_riff_of_no


def get_measure_length(bpm):
    return 60 / bpm * 4


def set_used_riff_num_info(phrases_dict, riffs_dict, modified_riffs_dict):
    for i in range(len(phrases_dict['rhythm_guitar_phrase'])):
        used_no = []
        phrase_info = phrases_dict['rhythm_guitar_phrase'][i]
        for used_riff in phrase_info['riffs']:
            # print(used_riff.keys())
            if not used_riff['modified']:
                for reference_riff in riffs_dict['griff']:
                    if parse_griff_json(used_riff) == parse_griff_json(reference_riff):
                        riff_no_info = {
                            'modified': False,
                            'no': reference_riff['no'],
                            'display': 'R'
                        }
                        used_no.append(riff_no_info)
            else:
                for reference_riff in modified_riffs_dict['griff']:
                    if parse_modified_griff_json(used_riff) == parse_modified_griff_json(reference_riff):
                        riff_no_info = {
                            'modified': True,
                            'no': reference_riff['no'],
                            'display': 'M'
                        }
                        used_no.append(riff_no_info)
        phrase_info['riffs_no'] = used_no
        phrase_info['raw_riffs_no'] = ' '.join([riff_no_info['display']+str(riff_no_info['no']) for riff_no_info in used_no])
        phrases_dict['rhythm_guitar_phrase'][i] = phrase_info

    for i in range(len(phrases_dict['rhythm_bass_phrase'])):
        used_no = []
        phrase_info = phrases_dict['rhythm_bass_phrase'][i]
        for used_riff in phrase_info['riffs']:
            if not used_riff['modified']:
                for reference_riff in riffs_dict['briff']:
                    if parse_briff_json(used_riff) == parse_briff_json(reference_riff):
                        riff_no_info = {
                            'modified': False,
                            'no': reference_riff['no'],
                            'display': 'R'
                        }
                        used_no.append(riff_no_info)
            else:
                for reference_riff in modified_riffs_dict['briff']:
                    if parse_modified_briff_json(used_riff) == parse_modified_briff_json(reference_riff):
                        riff_no_info = {
                            'modified': True,
                            'no': reference_riff['no'],
                            'display': 'M'
                        }
                        used_no.append(riff_no_info)
        phrase_info['riffs_no'] = used_no
        phrase_info['raw_riffs_no'] = ' '.join([riff_no_info['display']+str(riff_no_info['no']) for riff_no_info in used_no])
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
        if used_riff[0] == 'R':
            used_riffs.append({'modified': False, 'no': int(used_riff[1:]), 'display': 'R'})
        if used_riff[0] == 'M':
            used_riffs.append({'modified': True, 'no': int(used_riff[1:]), 'display': 'M'})
    return used_riffs


def get_rhythm_arrangements_from_raw(raw_arrangements):
    arrangements = []

    for arrangement in raw_arrangements.split('; '):
        riff_index = int(arrangement.split(' ')[0])
        degree = arrangement.split(' ')[1]
        if get_relative_distance(degree) is None:
            raise Exception()
        else:
            arrangements.append([riff_index, degree])
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


def refresh_rhythm_riff_info(phrase_info, phrase_type, riff_dict, modified_riff_dict):

    according_riff_dict = {
        'rhythm_guitar_phrase': 'griff',
        'rhythm_bass_phrase': 'briff',
    }

    riff_list = []

    for riff_no in phrase_info['riffs_no']:
        if riff_no['modified'] is False:
            riff = get_riff_of_no(riff_dict, according_riff_dict[phrase_type], riff_no['no'])
            riff['modified'] = False
            riff_list.append(riff)

        else:
            riff = get_riff_of_no(modified_riff_dict, according_riff_dict[phrase_type], riff_no['no'])
            riff['modified'] = True
            riff_list.append(riff)

    phrase_info['riffs'] = riff_list


def refresh_drum_riff_info(phrase_info, phrase_type, riff_dict):
    according_riff_dict = {
        'drum_phrase': 'driff'
    }
    riff_list = []
    for riff_no in phrase_info['riffs_no']:
        riff = get_riff_of_no(riff_dict, according_riff_dict[phrase_type], riff_no)
        riff_list.append(riff)

    phrase_info['riffs'] = riff_list


def refresh_all_phrases(phrases, riffs, modified_riffs):
    for phrase_type in ['rhythm_guitar_phrase', 'rhythm_bass_phrase']:
        phrases_list = phrases[phrase_type]
        for phrase in phrases_list:
            refresh_rhythm_riff_info(phrase, phrase_type, riffs, modified_riffs)
    for phrase_type in ['drum_phrase']:
        phrases_list = phrases[phrase_type]
        for phrase in phrases_list:
            refresh_drum_riff_info(phrase, phrase_type, riffs)


def get_phrase_of_no(phrase_dict, phrase_type, no):
    for phrase_info in phrase_dict[phrase_type]:
        current_no = phrase_info['no']
        if current_no == no:
            return phrase_info
