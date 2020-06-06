import pretty_midi
from music21 import analysis
import numpy as np
import scipy.stats as stats
import math


class MusicFragment:
    def __init__(self, path):
        self.path = path
        self.pm = pretty_midi.PrettyMIDI(path)
        self.length = self.pm.get_end_time()
        self.measures_num = math.ceil(self.length / 2.0)

    def get_note_lengths_divided_by_measure(self):
        notes_length = [[0 for _ in range(12)] for _ in range(self.measures_num)]

        for instr in self.pm.instruments:
            for note in instr.notes:
                pitch = note.pitch

                start_time, end_time = note.start, note.end
                start_measure, end_measure = int(start_time // 2), int(end_time // 2)
                if math.modf(end_time / 2)[0] == 0:
                    end_measure -= 1

                if start_measure == end_measure:
                    notes_length[start_measure][pitch % 12] += end_time - start_time

                else:
                    # start_measure
                    notes_length[start_measure][pitch % 12] += (start_measure+1) * 2.0 - start_time

                    # measures in between
                    for measure in range(start_measure+1, end_measure):
                        notes_length[measure][pitch % 12] += 2.0

                    # end_measure
                    notes_length[end_measure][pitch % 12] += end_time - end_measure * 2.0

        return notes_length

    def crop_by_measure(self):
        cropped_pm = []

        for measure in range(self.measures_num):
            pm = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(program=0)

            for instr in self.pm.instruments:
                for note in instr.notes:
                    pitch = note.pitch
                    velocity = note.velocity

                    start_time, end_time = note.start, note.end
                    start_measure, end_measure = int(start_time // 2), int(end_time // 2)

                    if start_measure > measure or end_measure < measure:
                        continue

                    else:
                        if start_measure == end_measure == measure:
                            new_start, new_end = start_time - measure*2.0, end_time - measure*2.0
                            new_note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                                        start=new_start, end=new_end)
                            piano.notes.append(new_note)

                        elif start_measure == measure < end_measure:
                            new_start, new_end = start_time - measure*2.0, 2.0
                            new_note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                                        start=new_start, end=new_end)
                            piano.notes.append(new_note)

                        elif start_measure < measure == measure:
                            new_start, new_end = 0, end_time - measure*2.0
                            new_note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                                        start=new_start, end=new_end)
                            piano.notes.append(new_note)

                        else:
                            assert start_measure < measure < end_measure
                            new_start, new_end = 0, 2.0
                            new_note = pretty_midi.Note(velocity=velocity, pitch=pitch,
                                                        start=new_start, end=new_end)
                            piano.notes.append(new_note)

            pm.instruments.append(piano)
            cropped_pm.append(pm)

        return cropped_pm

    def get_note_lengths(self):
        notes_length = [0 for _ in range(12)]
        for instr in self.pm.instruments:
            if not instr.is_drum:
                for note in instr.notes:
                    length = note.end - note.start
                    pitch = note.pitch
                    notes_length[pitch % 12] += length

        return notes_length

    def tonality_by_measures(self):
        measures_tonality = [None for _ in range(self.measures_num)]

        for measure in range(self.measures_num):
            note_lengths_of_measure = self.get_note_lengths_divided_by_measure()[measure]
            tonality = krumhansl_schmuckler(note_lengths_of_measure)
            measures_tonality[measure] = tonality

        return measures_tonality


def get_weights(mode, name='ks'):
    if name == 'kk':
        a = analysis.discrete.KrumhanslKessler()
        # Strong tendancy to identify the dominant key as the tonic.
    elif name == 'ks':
        a = analysis.discrete.KrumhanslSchmuckler()
    elif name == 'ae':
        a = analysis.discrete.AardenEssen()
        # Weak tendancy to identify the subdominant key as the tonic.
    elif name == 'bb':
        a = analysis.discrete.BellmanBudge()
        # No particular tendancies for confusions with neighboring keys.
    elif name == 'tkp':
        a = analysis.discrete.TemperleyKostkaPayne()
        # Strong tendancy to identify the relative major as the tonic in minor keys. Well-balanced for major keys.
    else:
        assert name == 's'
        a = analysis.discrete.SimpleWeights()
        # Performs most consistently with large regions of music, becomes noiser with smaller regions of music.
    return a.getWeights(mode)


def krumhansl_schmuckler(note_lengths):
    key_profiles = [0 for _ in range(24)]

    for key_index in range(24):

        if key_index // 12 == 0:
            mode = 'major'
        else:
            mode = 'minor'
        weights = get_weights(mode, 'kk')

        current_note_length = note_lengths[key_index:] + note_lengths[:key_index]

        pearson = stats.pearsonr(current_note_length, weights)[0]

        key_profiles[key_index] = math.fabs(pearson)

    key_name = get_key_name(np.argmax(key_profiles))
    return key_name
    # print(key_profiles, '\n', note_lengths)


def get_key_name(index):
    if index // 12 == 0:
        mode = 'major'
    else:
        mode = 'minor'

    tonic_list = ['C', '♭D', 'D', '♭E', 'E', 'F', '♭g', 'G', '♭A', 'A', '♭B', 'B']
    tonic = tonic_list[index % 12]
    return {'Tonic': tonic, 'Mode': mode}
