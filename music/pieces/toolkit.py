from music.custom_elements.riff import parse_griff_json, parse_briff_json
from music.custom_elements.drum_riff import parse_driff_json


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
