from music.pieces.track.track import *
from music.pieces.phrase.rhythm_phrase import *
from music.pieces.phrase.drum_phrase import *

from music.pieces.phrase.toolkit import set_used_riff_num_info
from music.pieces.track.toolkit import set_used_phrase_num_info

import pretty_midi
from music.process.audio_related import *
from dataset.web_db import get_song_table
import os


class Song:
    def __init__(self, name):
        self.name = name
        self.songwriter = ''
        self.genre = ''

        self.tracks = []
        self.excluded_tracks_index = []
        self.pm = None

        self.save_dir = 'D:/PycharmProjects/RiffGAN/data/pieces/songs/'
        self.midi_path = self.save_dir + 'midi/' + self.name + '.mid'
        self.json_path = self.save_dir + 'json/' + self.name + '.json'
        self.wav_path = self.save_dir + 'audio/' + self.name + '.wav'

    def set_writer(self, writer):
        self.songwriter = writer

    def set_genre(self, genre):
        self.genre = genre

    def set_excluded_tracks_index(self, excluded_tracks_index):
        self.excluded_tracks_index = excluded_tracks_index

    def add_track(self, track):
        assert isinstance(track, Track)
        self.tracks.append(track)

    def set_tracks(self, tracks):
        self.tracks = tracks

    def add_tracks_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()
        for i, track in enumerate(self.tracks):
            if i in self.excluded_tracks_index:
                continue
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

    def play_with_no_init(self):
        assert os.path.exists(self.midi_path)
        play_music_without_init(self.midi_path)

    def export_wav(self):
        assert os.path.exists(self.midi_path)
        export_as_wav(self.midi_path, self.wav_path)

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.export_json_dict(), f)

    def save_to_db(self):
        song_table = get_song_table()
        song_info = self.export_json_dict()

        riffs = self.get_all_riffs()
        phrases = self.get_all_phrases()
        tracks = self.get_all_tracks()

        set_used_riff_num_info(phrases, riffs)
        set_used_phrase_num_info(tracks, phrases)

        song_info['riffs'] = riffs
        song_info['phrases'] = phrases
        song_info['tracks'] = tracks

        if song_table.find_one({'name': self.name}) is None:
            song_table.insert_one(song_info)
        else:
            song_table.update_one(
                {'name': self.name},
                {'$set': song_info}
            )

    def export_json_dict(self):
        info_dict = {
            "name": self.name,
            'songwriter': self.songwriter,
            'genre': self.genre,
            "tracks": [track.export_json_dict() for track in self.tracks],
            'excluded_track_index': self.excluded_tracks_index
        }
        return info_dict

    def get_all_riffs(self):
        riffs_dict = {
            'griff': [],
            'briff': [],
            'driff': []
        }

        for track in self.tracks:
            for phrase in track.phrases:
                if track.is_drum:
                    assert isinstance(phrase, DrumPhrase)
                    for riff in phrase.riffs:
                        if riff not in [parse_driff_json(info) for info in riffs_dict['driff']]:
                            driff_info = riff.export_json_dict()
                            driff_info['no'] = len(riffs_dict['driff']) + 1
                            riffs_dict['driff'].append(driff_info)
                else:
                    assert isinstance(phrase, RhythmPhrase)
                    if phrase.instr_type == 'guitar':
                        for riff in phrase.riffs:
                            if riff not in [parse_griff_json(info) for info in riffs_dict['griff']]:
                                griff_info = riff.export_json_dict()
                                griff_info['no'] = len(riffs_dict['griff']) + 1
                                griff_info['raw_degrees_and_types'] = riff.get_degrees_and_types_str()
                                griff_info['raw_timestamps'] = riff.get_timestamps_str()
                                riffs_dict['griff'].append(griff_info)
                    else:
                        assert phrase.instr_type == 'bass'
                        for riff in phrase.riffs:
                            if riff not in [parse_briff_json(info) for info in riffs_dict['briff']]:
                                briff_info = riff.export_json_dict()
                                briff_info['no'] = len(riffs_dict['briff']) + 1
                                briff_info['raw_degrees_and_types'] = riff.get_degrees_and_types_str()
                                briff_info['raw_timestamps'] = riff.get_timestamps_str()
                                riffs_dict['briff'].append(briff_info)
        return riffs_dict

    def get_all_phrases(self):
        phrases_dict = {
            'rhythm_guitar_phrase': [],
            'rhythm_bass_phrase': [],
            'drum_phrase': []
        }

        for track in self.tracks:
            for phrase in track.phrases:
                if track.is_drum:
                    assert isinstance(phrase, DrumPhrase)
                    if phrase not in [parse_drum_phrase_json(info) for info in phrases_dict['drum_phrase']]:
                        phrase_info = phrase.export_json_dict()

                        phrase_info['no'] = len(phrases_dict['drum_phrase']) + 1
                        phrase_info['raw_arrangements'] = phrase.get_arrangement_str()
                        phrases_dict['drum_phrase'].append(phrase_info)
                else:
                    assert isinstance(phrase, RhythmPhrase)

                    if phrase.instr_type == 'guitar':
                        if phrase not in [parse_rhythm_phrase_json(info) for info in phrases_dict['rhythm_guitar_phrase']]:
                            phrase_info = phrase.export_json_dict()

                            phrase_info['no'] = len(phrases_dict['rhythm_guitar_phrase']) + 1
                            phrase_info['raw_arrangements'] = phrase.get_arrangement_str()
                            phrases_dict['rhythm_guitar_phrase'].append(phrase_info)

                    else:
                        assert phrase.instr_type == 'bass'
                        if phrase not in [parse_rhythm_phrase_json(info) for info in phrases_dict['rhythm_bass_phrase']]:
                            phrase_info = phrase.export_json_dict()

                            phrase_info['no'] = len(phrases_dict['rhythm_bass_phrase']) + 1
                            phrase_info['raw_arrangements'] = phrase.get_arrangement_str()
                            phrases_dict['rhythm_bass_phrase'].append(phrase_info)
        return phrases_dict

    def get_all_tracks(self):
        tracks_info = []

        for track in self.tracks:
            track_info = track.export_json_dict()
            if track.is_drum:
                track_info['raw_bpm_info'] = track.get_bpm_info_str()
                track_info['raw_arrangements'] = track.get_arrangement_str()
            else:
                if track.instr_type == 'guitar':
                    track_info['raw_bpm_info'] = track.get_bpm_info_str()
                    track_info['raw_arrangements'] = track.get_arrangement_str()
                    track_info['raw_tonality_info'] = track.get_tonality_info_str()
                else:
                    assert track.instr_type == 'bass'
                    track_info['raw_bpm_info'] = track.get_bpm_info_str()
                    track_info['raw_arrangements'] = track.get_arrangement_str()
                    track_info['raw_tonality_info'] = track.get_tonality_info_str()

            tracks_info.append(track_info)

        return tracks_info


def create_song_from_json(path):
    with open(path, 'r') as f:
        song_info = json.loads(f.read())
        return parse_song_json(song_info)


def load_song_from_db(name):
    song_table = get_song_table()

    song = song_table.find_one({'name': name})
    if song is None:
        raise Exception(f'No song found in name of {name}')
    else:
        return song


def get_empty_song():
    info_dict = {
        "name": '',
        'songwriter': '',
        'genre': '',
        "tracks": [],
        'excluded_track_index': [],
        'phrases': {
            'rhythm_guitar_phrase': [],
            'rhythm_bass_phrase': [],
            'drum_phrase': []
        },
        'riffs': {
            'griff': [],
            'driff': [],
            'briff': []
        }
    }
    return info_dict


def parse_song_json(song_info):
    song = Song(song_info['name'])
    song.set_writer(song_info['songwriter'])
    song.set_genre(song_info['genre'])
    song.set_excluded_tracks_index(song_info['excluded_track_index'])
    song.set_tracks([parse_track_json(track_info) for track_info in song_info['tracks']])

    return song
