import pretty_midi
import numpy as np
from music.custom_elements.rhythm_riff.toolkit import *


def generate_midi_segment_from_tensor(data, path, bpm=120):
    pm = pretty_midi.PrettyMIDI()
    instr_track = pretty_midi.Instrument(program=0, is_drum=False, name='Instr')
    sixty_fourth_length = 60 / bpm / 16
    note_range = 84
    time_step = 64

    for note in range(note_range):
        during_note = False
        note_begin = 0
        for time in range(time_step):
            has_note = data[time, note] >= 0.5

            if has_note:
                if not during_note:
                    during_note = True
                    note_begin = time * sixty_fourth_length
                else:
                    if time != time_step - 1:
                        continue
                    else:
                        note_end = time * sixty_fourth_length
                        instr_track.notes.append(pretty_midi.Note(127, note + 24, note_begin, note_end))
                        during_note = False
            else:
                if not during_note:
                    continue
                else:
                    note_end = time * sixty_fourth_length
                    instr_track.notes.append(pretty_midi.Note(127, note + 24, note_begin, note_end))
                    during_note = False
    pm.instruments.append(instr_track)
    pm.write(path)


def save_midis(bars, path, instr_type):
    from util.npy_related import plot_data
    pm = pretty_midi.PrettyMIDI()

    if instr_type == 'guitar':
        note_range = (36, 96)
        instr_num = 29
        # standard tune: [E2, D6] -> [C2, C7)
    else:
        assert instr_type == 'bass'
        note_range = (24, 72)
        instr_num = 33
        # standard tune: [E1, G4] -> [C1, C5)

    padded_bars = bars
    padded_bars = padded_bars.reshape((-1, padded_bars.shape[1], padded_bars.shape[2], padded_bars.shape[3]))
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[1]):
        padded_bars_list.append(padded_bars[:, ch_idx, :, :].reshape(padded_bars.shape[0],
                                                                     padded_bars.shape[2],
                                                                     padded_bars.shape[3]))

    pianoroll = padded_bars_list[0]
    pianoroll = pianoroll.reshape((pianoroll.shape[0] * pianoroll.shape[1], pianoroll.shape[2]))
    pianoroll_diff = np.concatenate((np.zeros((1, note_range[1]-note_range[0]), dtype=int), pianoroll,
                                     np.zeros((1, note_range[1]-note_range[0]), dtype=int)))
    pianoroll_search = np.diff(pianoroll_diff.astype(int), axis=0)

    instrument = pretty_midi.Instrument(program=instr_num, is_drum=False, name='Instr')

    tempo = 120
    beat_resolution = 16

    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 16
    phrase_end_time = 60.0 / tempo * 4 * pianoroll.shape[0]

    for note_num in range(0, note_range[1]-note_range[0]):
        start_idx = (pianoroll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))

        end_idx = (pianoroll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    try:
                        if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                            t.append(j)
                    except:
                        print(len(temp_start_time), j)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])

        start_time = temp_start_time
        end_time = temp_end_time
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                note = pretty_midi.Note(velocity=127, pitch=get_nearest_in_tone_note(note_num+note_range[0]), start=start_time[idx],
                                        end=end_time[idx])
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    note = pretty_midi.Note(velocity=127, pitch=get_nearest_in_tone_note(note_num+note_range[0]), start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    note = pretty_midi.Note(velocity=127, pitch=get_nearest_in_tone_note(note_num+note_range[0]), start=start_time[idx],
                                            end=phrase_end_time)
                instrument.notes.append(note)
    instrument.notes.sort(key=lambda note: note.start)

    pm.instruments.append(instrument)
    pm.write(path)


def get_nearest_in_tone_note(note, tonality=('C', 'major')):
    tonic, mode = tonality
    if mode == 'major':
        note_arrange = [0, 2, 4, 5, 7, 9, 11]
    else:
        assert mode == 'minor'
        note_arrange = [0, 2, 3, 5, 7, 8, 10]

    root_dist = note_name_to_num(tonic + '1') % 12

    relative_dist = (note - root_dist) % 12

    min_gap = 12
    for n in note_arrange:
        if abs(n - relative_dist) < abs(min_gap):
            min_gap = n - relative_dist

    return note + min_gap


if __name__ == '__main__':
    print(get_nearest_in_tone_note(15))