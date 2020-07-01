
def get_chord(degree_and_type):
    degree, type = degree_and_type
    distance = get_relative_distance(degree)
    chord_pattern = get_chord_pattern(type)
    return [note + distance for note in chord_pattern]


def get_relative_distance(degree, mode='major'):
    major_degree_dict = {'I': 0, 'II': 2, 'III': 4, 'IV': 5, 'V': 7, 'VI': 9, 'VII': 11}
    minor_degree_dict = {'I': 0, 'II': 2, 'III': 3, 'IV': 5, 'V': 7, 'VI': 8, 'VII': 10}

    if mode == 'major':
        degree_dict = major_degree_dict
    else:
        assert mode == 'minor'
        degree_dict = minor_degree_dict

    octave = 0
    if '<' in degree:
        octave = -degree.count('<')
    elif '>' in degree:
        octave = degree.count('>')

    alt = 0
    if '+' in degree:
        alt = 1
    elif '-' in degree:
        alt = -1

    clean_degree = degree[abs(octave):]
    if alt != 0:
        clean_degree = clean_degree[:-1]

    try:
        return degree_dict[clean_degree] + octave * 12 + alt
    except:
        return None

    # print(clean_degree, octave, alt)


def get_measure_length(bpm):
    return 60 / bpm * 4


def note_name_to_num(name):
    major_degree_dict = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    octave = int(name[-1])
    note_name = name[:-1]
    if len(note_name) == 1:
        note_num = major_degree_dict[note_name]
    else:
        alt_str, raw_note = note_name[0], note_name[1]
        if alt_str == '-':
            alt = -1
        else:
            assert alt_str == '+'
            alt = 1
        note_num = major_degree_dict[raw_note] + alt
    return note_num + 12 * (octave+1)


def get_chord_pattern(chord_type):
    chord = None
    # Others
    if chord_type in ['']:
        # Single note
        chord = [0]

    if chord_type in ['5']:
        # Power
        # root | perfect
        chord = [0, 7]

    # Triad
    if chord_type in ['M', 'maj', 'Δ']:
        # Major triad
        # root | major | perfect
        chord = [0, 4, 7]
    if chord_type in ['m', 'min', '-']:
        # Minor triad
        # root | minor | perfect
        chord = [0, 3, 7]
    if chord_type in ['aug', '+']:
        # Augmented triad
        # root | major | augmented
        chord = [0, 4, 8]
    if chord_type in ['dim', 'o']:
        # Diminished triad
        # root | minor | diminished
        chord = [0, 3, 6]

    # Seventh
    if chord_type in ['o7', 'dim7']:
        # Diminished seventh
        # root | minor | diminished | diminished
        chord = [0, 3, 6, 9]
    if chord_type in ['ø7', '7♭5']:
        # Half-diminished seventh
        # root | minor | diminished | minor
        chord = [0, 3, 6, 10]
    if chord_type in ['m7', 'min7', '-7']:
        # Minor seventh
        # root | minor | perfect | minor
        chord = [0, 3, 7, 10]
    if chord_type in ['mM7', 'mmaj7', '−Δ7', '−M7']:
        # Minor major seventh
        # root | minor | perfect | major
        chord = [0, 3, 7, 11]
    if chord_type in ['7', 'dom7']:
        # Dominant seventh
        # root | major | perfect | minor
        chord = [0, 4, 7, 10]
    if chord_type in ['M7', 'maj7', 'Δ7']:
        # Major seventh
        # root | major | perfect | major
        chord = [0, 4, 7, 11]
    if chord_type in ['+7', 'aug7', '7+', '7+5', '7♯5']:
        # Augmented seventh
        # root | major | augmented | minor
        chord = [0, 4, 8, 10]
    if chord_type in ['+M7', 'M7+5', '+Δ7', 'M7♯5']:
        # Augmented major seventh
        # root | major | augmented | major
        chord = [0, 4, 8, 11]

    # Extended
    if chord_type in ['9']:
        # Dominant ninth
        # dominant seventh + major ninth
        chord = [0, 4, 7, 10, 14]
    if chord_type in ['11']:
        # Dominant eleventh
        # dominant seventh + major ninth + perfect eleventh
        chord = [0, 4, 7, 10, 14, 17]
    if chord_type in ['13']:
        # Dominant thirteenth
        # dominant seventh + perfect eleventh + major thirteenth
        chord = [0, 4, 7, 10, 14, 17, 21]

    # Altered
    # Seventh augmented fifth = Augmented seventh
    if chord_type in ['7−9', '7♭9']:
        # Seven minor ninth
        # dominant seventh + minor ninth
        chord = [0, 4, 7, 10, 13]
    if chord_type in ['7+9', '7♯9']:
        # Seventh sharp ninth
        # dominant seventh + augmented ninth
        chord = [0, 4, 7, 10, 15]
    if chord_type in ['7+11', '7♯11']:
        # Seventh augmented eleventh
        # dominant seventh + augmented eleventh
        chord = [0, 4, 7, 10, 14, 19]
    if chord_type in ['7−13', '7♭13']:
        # Seventh diminished thirteenth
        # dominant seventh + minor thirteenth
        chord = [0, 4, 7, 10, 14, 18, 20]
    if chord_type in ['ø', 'ø7', 'm7♭5']:
        # Half-diminished seventh
        # minor seventh + diminished fifth
        chord = [0, 3, 6, 10]

    # Added tone
    if chord_type in ['2', 'add9']:
        # Add nine
        # major triad + major ninth
        chord = [0, 2, 4, 7]
    if chord_type in ['4', 'add11']:
        # Add fourth
        # major triad + perfect fourth
        chord = [0, 4, 5, 7]
    if chord_type in ['6']:
        # Add sixth
        # major triad + major sixth
        chord = [0, 4, 5, 7, 9]
    if chord_type in ['6/9']:
        # Six-nine
        # major triad + major sixth + major ninth
        chord = [0, 2, 4, 7, 9]
    if chord_type in ['7/6']:
        # Seven-six
        # major triad + major sixth + minor seventh
        chord = [0, 4, 7, 9, 10]

    # Suspended chords
    if chord_type in ['sus2']:
        # Suspended second
        # open fifth + major second
        chord = [0, 2, 7]
    if chord_type in ['sus4']:
        # Suspended fourth
        # open fifth + perfect fourth
        chord = [0, 5, 7]
    if chord_type in ['9sus4']:
        # Jazz sus
        # open fifth + perfect fourth + minor seventh + major ninth
        chord = [0, 2, 5, 7, 10]

    return chord


def get_degrees_and_types_from_raw(raw_degrees_and_types):
    degrees_and_types = []

    for degree_and_type in raw_degrees_and_types.split('; '):

        if len(degree_and_type.split(' ')) == 1:
            degree = degree_and_type.split(' ')[0]
            if get_relative_distance(degree) is None:
                raise Exception()
            else:
                degrees_and_types.append([degree, ''])
        else:
            degree = degree_and_type.split(' ')[0]
            chord_type = degree_and_type.split(' ')[1]
            if get_relative_distance(degree) is None or get_chord_pattern(chord_type) is None:
                raise Exception()
            else:
                degrees_and_types.append([degree, chord_type])
    return degrees_and_types


def get_timestamps_from_raw(raw_time_stamps):
    timestamps = []
    for timestamp in raw_time_stamps.split(' '):
        timestamps.append(float(timestamp))
    return timestamps


def get_largest_num_of_json(json_dict):
    max_num = 0
    for info in json_dict:
        current_no = info['no']
        if current_no > max_num:
            max_num = current_no
    return max_num


def time_stamps_convert(simple_ts, bpm):
    detailed_ts = []
    current_start = 0.0
    unit_length = 60 / bpm

    for ts in simple_ts:
        start = current_start
        end = ts * unit_length + start
        detailed_ts.append((start, end))

        current_start = end

    return detailed_ts


def get_guitar_str(code):
    assert code in range(24, 32)
    info_dict = {
        24: 'Acoustic Guitar (nylon)',
        25: 'Acoustic Guitar (steel)',
        26: 'Electric Guitar (jazz)',
        27: 'Electric Guitar (clean)',
        28: 'Electric Guitar (muted)',
        29: 'Overdriven Guitar',
        30: 'Distortion Guitar',
        31: 'Guitar harmonics'
    }
    return info_dict[code]


def get_bass_str(code):
    assert code in range(32, 40)
    info_dict = {
        32: 'Acoustic Bass',
        33: 'Electric Bass (finger)',
        34: 'Electric Bass (pick)',
        35: 'Fretless Bass',
        36: 'Slap Bass 1',
        37: 'Slap Bass 2',
        38: 'Synth Bass 1',
        39: 'Synth Bass 2'
    }
    return info_dict[code]


def get_all_used_riffs(phrase_dict, phrase_type):
    used_riffs_no = []
    for phrase_info in phrase_dict[phrase_type]:
        used_riffs = phrase_info['riffs_no']
        for riff_no in used_riffs:
            if riff_no not in used_riffs_no:
                used_riffs_no.append(riff_no)
    return used_riffs_no


def get_all_used_phrases(track_dict, track_type):
    used_riffs_no = []
    track_type_transform = {
        'drum': {
            'is_drum': True,
            'instr_type': None
        },
        'guitar': {
            'is_drum': False,
            'instr_type': 'guitar'
        },
        'bass': {
            'is_drum': False,
            'instr_type': 'bass'
        }
    }
    for track_info in track_dict:
        if track_info['is_drum'] == track_type_transform[track_type]['is_drum'] and track_info['instr_type'] == track_type_transform[track_type]['instr_type']:
            used_phrases = track_info['phrases_no']
            for riff_no in used_phrases:
                if riff_no not in used_riffs_no:
                    used_riffs_no.append(riff_no)
    return used_riffs_no


def get_riff_of_no(riff_dict, riff_type, no):
    for riff_info in riff_dict[riff_type]:
        current_no = riff_info['no']
        if current_no == no:
            return riff_info


if __name__ == '__main__':
    # get_degrees_and_types_from_raw('I 5; II 5')
    print(note_name_to_num('ae'))