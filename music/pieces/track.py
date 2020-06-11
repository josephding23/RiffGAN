from music.custom_elements.riff import *
from music.pieces.toolkit import *
from music.pieces.phrase import Phrase, RhythmPhrase, SoloPhrase, DrumPhrase
from music.custom_elements.drum_riff import *
import pretty_midi


class Track:
    def __init__(self, name, bpm_list, tonality_list, is_drum=False, is_rhythm=True):
        self.name = name

        self.phrases = []

        self.bpm_list = bpm_list
        self.tonality_list = tonality_list

        self.is_drum = is_drum
        self.is_rhythm = is_rhythm

        self.pm = None
        self.save_path = '../../data/pieces/tracks/'

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

    def get_phrases_num(self):
        print(len(self.phrases))

    def add_phrases_to_pm(self):
        if self.is_drum:
            self.add_drum_phrases_to_pm()
        elif self.is_rhythm:
            self.add_rhythm_phrases_to_pm()

    def add_drum_phrases_to_pm(self):

        self.pm = pretty_midi.PrettyMIDI()
        drum = pretty_midi.Instrument(program=0, name=self.name, is_drum=True)

        for phrase in self.phrases:
            assert isinstance(phrase, DrumPhrase)

            phrase_start = self.get_measure_start_time(phrase.start_measure)
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
                                drum.notes.append(note)

                riff_start += length_per_measure * riff.measure_length

        self.pm.instruments.append(drum)

    def add_rhythm_phrases_to_pm(self):
        self.pm = pretty_midi.PrettyMIDI()
        instr = pretty_midi.Instrument(program=0, name=self.name)

        for phrase in self.phrases:
            assert isinstance(phrase, RhythmPhrase)
            instr.program = phrase.instr
            phrase_start = self.get_measure_start_time(phrase.start_measure)
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

    def save(self, name):
        self.pm.write(self.save_path + name)
