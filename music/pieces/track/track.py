from music.pieces.phrase.drum_phrase import *
from music.pieces.phrase.rhythm_phrase import *
import pretty_midi


class Track:
    def __init__(self, name, bpm_list, tonality_list, is_drum, instr_type):
        self.name = name

        self.phrases = []
        self.arrangement = []

        self.bpm_list = bpm_list
        self.tonality_list = tonality_list

        self.is_drum = is_drum
        self.instr_type = instr_type
        self.is_rhythm = True

        self.pm = None
        self.save_dir = '../data/pieces/tracks/'
        self.midi_path = ''

    def get_measure_start_time(self, measure):
        start_time = 0
        if len(self.bpm_list) == 1:
            assert self.bpm_list[0][0] == 0
            start_time = get_measure_length(self.bpm_list[0][1]) * measure
        elif measure > self.bpm_list[-1][0]:
            for i in range(len(self.bpm_list)-1):
                start_measure, bpm = self.bpm_list[i]
                next_measure = self.bpm_list[i+1][0]

                start_time += (next_measure - start_measure) * get_measure_length(bpm)

            start_time += (measure - self.bpm_list[-1][0]) * get_measure_length(self.bpm_list[-1][1])

        else:
            for i in range(len(self.bpm_list)-1):
                start_measure, bpm = self.bpm_list[i]
                next_measure, _ = self.bpm_list[i + 1]

                if measure <= next_measure:
                    start_time += (measure - start_measure) * get_measure_length(bpm)

                else:
                    start_time += (next_measure - start_measure) * get_measure_length(bpm)
        return start_time

    def add_phrase(self, phrase):
        assert isinstance(phrase, Phrase)
        self.phrases.append(phrase)

    def set_phrases(self, phrases):
        self.phrases = phrases

    def set_arrangement(self, arrangement):
        self.arrangement = arrangement

    def get_phrases_num(self):
        print(len(self.phrases))

    def add_phrases_to_pm(self):
        if self.is_drum:
            self.add_drum_phrases_to_pm()
        else:
            self.add_rhythm_phrases_to_pm()

    def add_drum_phrases_to_pm(self):

        self.pm = pretty_midi.PrettyMIDI()
        drum = pretty_midi.Instrument(program=0, name=self.name, is_drum=True)

        for phrase_num, start_measure in self.arrangement:
            phrase = self.phrases[phrase_num]
            assert isinstance(phrase, DrumPhrase)

            phrase_start = self.get_measure_start_time(start_measure)
            riff_start = phrase_start
            length_per_measure = get_measure_length(phrase.bpm)

            for arrange in phrase.arrangement:
                riff = phrase.riffs[arrange]
                for part, pattern in riff.patterns.items():
                    if pattern is '':
                        continue
                    else:
                        assert isinstance(pattern, str)

                        total_num = len(pattern)
                        measure_length = get_measure_length(phrase.bpm) * riff.measure_length
                        unit_length = measure_length / total_num

                        for i in range(total_num):
                            symbol = pattern[i]
                            if symbol == '_':
                                continue
                            else:
                                start_time, end_time = i * unit_length, (i + 1) * unit_length
                                start_time += riff_start
                                end_time += riff_start

                                note = pretty_midi.Note(velocity=100, pitch=translate_symbol(part, symbol),
                                                        start=start_time, end=end_time)
                                drum.notes.append(note)

                riff_start += length_per_measure * riff.measure_length

        self.pm.instruments.append(drum)

    def add_rhythm_phrases_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()
        instr = pretty_midi.Instrument(program=0, name=self.name)

        for phrase_num, start_measure in self.arrangement:
            phrase = self.phrases[phrase_num]
            assert isinstance(phrase, RhythmPhrase)
            instr.program = phrase.instr
            phrase_start = self.get_measure_start_time(start_measure)
            riff_start = phrase_start
            length_per_measure = get_measure_length(phrase.bpm)

            for arrange in phrase.arrangement:
                riff, riff_root_name = phrase.riffs[arrange[0]], arrange[1]
                riff_root_dist = get_relative_distance(riff_root_name)

                real_time_stamps = time_stamps_convert(riff.time_stamps, phrase.bpm)
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
                        note = pretty_midi.Note(velocity=velocity, pitch=note_dist + phrase.root_note + riff_root_dist,
                                                start=start_time, end=end_time)
                        instr.notes.append(note)

                riff_start += length_per_measure * riff.measure_length

        self.pm.instruments.append(instr)

    def save_midi(self, name):
        self.midi_path = self.save_dir + 'midi/' + name + '.mid'
        self.pm.write(self.midi_path)

    def play_it(self):
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music(self.midi_path)

    def play_with_no_init(self):
        assert self.midi_path is not '' and os.path.exists(self.midi_path)
        play_music_without_init(self.midi_path)

    def get_arrangement_str(self):
        info_str = ''
        for arrangement in self.arrangement:
            info_str += str(arrangement[0]) + ' ' + str(arrangement[1]) + '; '
        return info_str[:-2]

    def get_bpm_info_str(self):
        info_str = ''
        for bpm_info in self.bpm_list:
            info_str += str(bpm_info[0]) + ' ' + str(bpm_info[1]) + '; '
        return info_str[:-2]

    def get_tonality_info_str(self):
        info_str = ''
        for tonality_info in self.tonality_list:
            info_str += str(tonality_info[0]) + ' ' + str(tonality_info[1][0]) + ' ' + str(tonality_info[1][1]) + '; '
        return info_str[:-2]

    def export_json_dict(self):
        info_dict = {
            "name": self.name,
            "bpm_list": self.bpm_list,
            "tonality_list": self.tonality_list,
            "instr_type": self.instr_type,
            "is_drum": self.is_drum,
            "phrases": [phrase.export_json_dict() for phrase in self.phrases],
            "arrangements": self.arrangement
        }

        return info_dict

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.export_json_dict(), f)


def create_track_from_json(path):
    with open(path, 'r') as f:
        track_info = json.loads(f.read())
        return parse_track_json(track_info)


def parse_track_json(track_info):
    is_drum = track_info['is_drum']
    if is_drum:
        phrases = [parse_drum_phrase_json(phrase) for phrase in track_info['phrases']]
    else:
        phrases = [parse_rhythm_phrase_json(phrase) for phrase in track_info['phrases']]

    track = Track(
        name=track_info['name'],
        bpm_list=track_info['bpm_list'],
        tonality_list=track_info['tonality_list'],
        is_drum=track_info['is_drum'],
        instr_type=track_info['instr_type']
    )
    track.set_phrases(phrases)
    track.set_arrangement(track_info['arrangements'])

    return track
