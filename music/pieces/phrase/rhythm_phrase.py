from music.pieces.phrase.phrase import *
from music.pieces.phrase.toolkit import *
from music.custom_elements.rhythm_riff.guitar_riff import *
from music.custom_elements.rhythm_riff.bass_riff import *
from music.custom_elements.modified_riff.modified_riff import *
from util.music_generate import *


class RhythmPhrase(Phrase):
    def __init__(self, length, tonality, bpm, instr, instr_type):
        Phrase.__init__(self, length, bpm)

        self.tonality = tonality
        self.tonic, self.mode = self.tonality[0], self.tonality[1]
        self.root_note = note_name_to_num(self.tonic)

        self.instr = instr
        self.instr_type = instr_type
        if instr_type == 'guitar':
            self.instr_str = get_guitar_str(self.instr)
        else:
            assert self.instr_type == 'bass'
            self.instr_str = get_bass_str(self.instr)

        self.riffs = []
        self.arrangement = []

    def __eq__(self, other):
        assert isinstance(other, RhythmPhrase)
        return self.tonality == other.tonality and self.root_note == other.root_note and \
               self.length == other.length and self.bpm == other.bpm and \
               self.instr == other.instr and self.instr_type == other.instr_type and \
               self.riffs == other.riffs and self.arrangement == other.arrangement

    def set_riffs(self, riffs):
        self.riffs = riffs

    def set_arrangement(self, arrangement):
        self.arrangement = arrangement

    def add_riffs_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()

        instr = pretty_midi.Instrument(program=self.instr)
        riff_start = 0
        length_per_measure = get_measure_length(self.bpm)

        for arrange in self.arrangement:
            riff, riff_root_name = self.riffs[arrange[0]], arrange[1]
            riff_root_dist = get_relative_distance(riff_root_name)

            if isinstance(riff, GuitarRiff) or isinstance(riff, BassRiff):
                real_time_stamps = time_stamps_convert(riff.time_stamps, self.bpm)
                for i in range(len(real_time_stamps)):
                    start_time, end_time = real_time_stamps[i]
                    start_time += riff_start
                    end_time += riff_start
                    if type(riff.velocity) == int:
                        velocity = riff.velocity
                    else:
                        assert type(riff.velocity) == list
                        velocity = riff.velocity[i]
                    chord = riff.chords[i]

                    for note_dist in chord:
                        note = pretty_midi.Note(velocity=velocity, pitch=note_dist + self.root_note + riff_root_dist,
                                                start=start_time, end=end_time)
                        instr.notes.append(note)
                riff_start += length_per_measure * riff.measure_length
            else:
                assert isinstance(riff, ModifiedGuitarRiff) or isinstance(riff, ModifiedBassRiff)
                add_notes_from_nonzeros_to_instr(riff.nonzeros, riff.shape, instr, self.bpm, riff_start)

                riff_start += length_per_measure * riff.original_riff.measure_length

            # print(riff_start)

        self.pm.instruments.append(instr)

    def get_arrangement_str(self):
        info_str = ''
        for arrangement in self.arrangement:
            info_str += str(arrangement[0]) + ' ' + arrangement[1] + '; '
        return info_str[:-2]

    def export_json_dict(self):
        info_dict = {
            "length": self.length,
            "tonality": self.tonality,
            "bpm": self.bpm,
            "instr": self.instr,
            "instr_type": self.instr_type,
            "instr_str": self.instr_str,
            "riffs": [riff.export_json_dict() for riff in self.riffs],
            "arrangements": self.arrangement
        }
        return info_dict

    def save_json(self, name):
        with open(self.save_dir + 'json/' + name + '.json', 'w') as f:
            json.dump(self.export_json_dict(), f)


def create_rhythm_phrase_from_json(path):
    with open(path, 'r') as f:
        phrase_info = json.loads(f.read())
        return parse_rhythm_phrase_json(phrase_info)


def parse_rhythm_phrase_json(phrase_info):
    instr_type = phrase_info['instr_type']
    if instr_type == 'guitar':
        riffs = []
        for riff_info in phrase_info['riffs']:

            if not riff_info['modified']:
                riff = parse_griff_json(riff_info)
            else:
                riff = parse_modified_griff_json(riff_info)

            # riff = parse_griff_json(riff_info)
            riffs.append(riff)
    else:
        assert instr_type == 'bass'
        riffs = []
        for riff_info in phrase_info['riffs']:
            print(riff_info)
            if not riff_info['modified']:
                riff = parse_briff_json(riff_info)
            else:
                riff = parse_modified_briff_json(riff_info)

            # riff = parse_briff_json(riff_info)
            riffs.append(riff)

    rhythm_phrase = RhythmPhrase(
        length=phrase_info['length'],
        tonality=phrase_info['tonality'],
        bpm=phrase_info['bpm'],
        instr=phrase_info['instr'],
        instr_type=instr_type,
    )

    rhythm_phrase.set_riffs(riffs)
    rhythm_phrase.set_arrangement(phrase_info['arrangements'])

    return rhythm_phrase
