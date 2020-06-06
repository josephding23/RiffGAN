import pretty_midi


class GuitarRiff:
    def __init__(self, length, root_note, degree_and_types, time_stamps):
        self.length = length
        self.root_note = root_note
        self.degree_and_types = degree_and_types
        self.time_stamps = time_stamps
        self.pm = pretty_midi.PrettyMIDI


def get_relative_distance(degree_and_type, mode='major'):
    degree, type = degree_and_type
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
    print(clean_degree, octave, alt)


def test_distance():
    dt1 = ('<<I', '5')
    get_relative_distance(dt1)


if __name__ == '__main__':
    test_distance()