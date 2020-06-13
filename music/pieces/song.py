from music.pieces.track import *
import pretty_midi
from music.process.audio_related import *
import os


class Song:
    def __init__(self, name):
        self.name = name
        self.tracks = []
        self.pm = None

        self.save_dir = '/PycharmProjects/RiffGAN/data/pieces/songs/'
        self.midi_path = self.save_dir + 'midi/' + self.name + '.mid'
        self.json_path = self.save_dir + 'json/' + self.name + '.json'
        self.wav_path = self.save_dir + 'audio/' + self.name + '.wav'

    def add_track(self, track):
        assert isinstance(track, Track)
        self.tracks.append(track)

    def set_tracks(self, tracks):
        self.tracks = tracks

    def add_tracks_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()
        for track in self.tracks:
            instr = pretty_midi.Instrument(program=0, name=track.name, is_drum=track.is_drum)

            if track.is_drum:
                for phrase_num, start_measure in track.arrangement:
                    phrase = track.phrases[phrase_num]
                    assert isinstance(phrase, DrumPhrase)

                    phrase_start = track.get_measure_start_time(start_measure)
                    riff_start = phrase_start
                    length_per_measure = get_measure_length(phrase.bpm)

                    for arrange in phrase.arrangement:
                        riff = phrase.riffs[arrange]
                        for part, pattern in riff.patterns.items():
                            if pattern is None:
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
                                        instr.notes.append(note)

                        riff_start += length_per_measure * riff.measure_length

            elif track.is_rhythm:
                for phrase_num, start_measure in track.arrangement:
                    phrase = track.phrases[phrase_num]
                    assert isinstance(phrase, RhythmPhrase)
                    instr.program = phrase.instr

                    phrase_start = track.get_measure_start_time(start_measure)
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
                                note = pretty_midi.Note(velocity=velocity,
                                                        pitch=note_dist + phrase.root_note + riff_root_dist,
                                                        start=start_time, end=end_time)
                                instr.notes.append(note)

                        riff_start += length_per_measure * riff.measure_length

            self.pm.instruments.append(instr)

    def save_midi(self):
        assert self.pm is not None

        self.pm.write(self.midi_path)

    def play_it(self):
        assert os.path.exists(self.midi_path)
        play_music(self.midi_path)

    def export_wav(self):
        assert os.path.exists(self.midi_path)
        export_as_wav(self.midi_path, self.wav_path)

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.export_json_dict(), f)

    def export_json_dict(self):
        info_dict = {
            "name": self.name,
            "tracks": [track.export_json_dict() for track in self.tracks]
        }
        return info_dict

    def get_all_riffs(self):
        riffs_dict = {
            'griff': [],
            'briff': [],
            'driff': []
        }
        griff_no = 0
        briff_no = 0
        driff_no = 0

        for track in self.tracks:
            for phrase in track.phrases:
                if track.is_drum:
                    assert isinstance(phrase, DrumPhrase)
                    for riff in phrase.riffs:
                        if riff not in [parse_driff_json(info) for info in riffs_dict['driff']]:
                            driff_info = riff.export_json_dict()
                            driff_info['num'] = driff_no + 1
                            riffs_dict['driff'].append(driff_info)
                            driff_no += 1
                else:
                    assert isinstance(phrase, RhythmPhrase)
                    if phrase.instr_type == 'guitar':
                        for riff in phrase.riffs:
                            if riff not in [parse_griff_json(info) for info in riffs_dict['griff']]:
                                griff_info = riff.export_json_dict()
                                griff_info['num'] = griff_no + 1
                                riffs_dict['griff'].append(griff_info)
                                griff_no += 1
                    else:
                        assert phrase.instr_type == 'bass'
                        for riff in phrase.riffs:
                            if riff not in [parse_briff_json(info) for info in riffs_dict['briff']]:
                                briff_info = riff.export_json_dict()
                                briff_info['num'] = briff_no + 1
                                riffs_dict['briff'].append(briff_info)
                                briff_no += 1
        return riffs_dict

    def get_all_phrases(self):
        phrases_dict = {
            'rhythm_phrases': [],
            'drum_phrases': []
        }
        rhy_phrase_num = 0
        drum_phrase_num = 0

        for track in self.tracks:
            for phrase in track.phrases:
                if track.is_drum:
                    assert isinstance(phrase, DrumPhrase)
                    if phrase not in [parse_drum_phrase_json(info) for info in phrases_dict['drum_phrases']]:
                        phrase_info = phrase.export_json_dict()
                        phrase_info['num'] = rhy_phrase_num + 1
                        phrases_dict['drum_phrases'].append(phrase_info)
                        rhy_phrase_num += 1
                else:
                    assert isinstance(phrase, RhythmPhrase)
                    if phrase not in [parse_rhythm_phrase_json(info) for info in phrases_dict['rhythm_phrases']]:
                        phrase_info = phrase.export_json_dict()
                        phrase_info['num'] = drum_phrase_num + 1
                        phrases_dict['rhythm_phrases'].append(phrase_info)
                        drum_phrase_num += 1
        return phrases_dict

    def get_all_tracks(self):
        return self.export_json_dict()['tracks']


def create_song_drom_json(path):
    with open(path, 'r') as f:
        song_info = json.loads(f.read())
        return parse_song_json(song_info)


def parse_song_json(song_info):
    song = Song(song_info['name'])
    song.set_tracks([parse_track_json(track_info) for track_info in song_info['tracks']])

    return song