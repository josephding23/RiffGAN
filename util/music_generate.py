import pretty_midi
import numpy as np
from music.custom_elements.rhythm_riff.toolkit import *
from util.npy_related import *


def add_notes_from_nonzeros_to_instr(nonzeros, shape, instr_track, bpm=120, start_time=0):
    # pm = pretty_midi.PrettyMIDI()
    data = np.zeros(shape, np.float_)
    for nonzero in nonzeros:
        # print(nonzero)
        data[(int(nonzero[0]), int(nonzero[1]), int(nonzero[2]))] = 1.0

    # plot_data(data[0, :, :], shape)

    if shape[2] == 60:
        note_range = (36, 96)
        # standard tune: [E2, D6] -> [C2, C7)
    else:
        assert shape[2] == 48
        note_range = (24, 72)
        # standard tune: [E1, G4] -> [C1, C5)

    # instr_track = pretty_midi.Instrument(program=0, is_drum=False, name='Instr')
    sixty_fourth_length = 60 / bpm / 16

    for note in range(shape[2]):
        during_note = False
        note_begin = 0
        for i in range(shape[0]):
            for time in range(shape[1]):
                has_note = data[i, time, note] >= 0.5

                whole_time = i * shape[1] + time

                if has_note:
                    if not during_note:
                        during_note = True
                        note_begin = start_time + whole_time * sixty_fourth_length
                    else:
                        if not ((i == shape[0] - 1) and (time == shape[1] - 1)):
                            continue
                        else:
                            note_end = start_time + whole_time * sixty_fourth_length
                            instr_track.notes.append(pretty_midi.Note(127, note + note_range[0], note_begin, note_end))
                            during_note = False
                else:
                    if not during_note:
                        continue
                    else:
                        note_end = start_time + whole_time * sixty_fourth_length
                        instr_track.notes.append(pretty_midi.Note(127, note + note_range[0], note_begin, note_end))
                        during_note = False


def save_midis(bars, path, instr_type, pitch_correct=True, quantize=True):
    from util.data_plotting import plot_data
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
                if quantize:
                    start, end = get_quantization_time(start_time[idx], end_time[idx])
                else:
                    start, end = start_time[idx], end_time[idx]

                if pitch_correct:
                    pitch = get_nearest_in_tone_note(note_num + note_range[0])
                else:
                    pitch = note_num + note_range[0]

                note = pretty_midi.Note(velocity=127, pitch=pitch, start=start, end=end)
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    if quantize:
                        start, end = get_quantization_time(start_time[idx], end_time[idx] + threshold)
                    else:
                        start, end = start_time[idx], end_time[idx] + threshold

                    if pitch_correct:
                        pitch = get_nearest_in_tone_note(note_num+note_range[0])
                    else:
                        pitch = note_num + note_range[0]

                    note = pretty_midi.Note(velocity=127, pitch=pitch, start=start, end=end)

                else:
                    if quantize:
                        start, end = get_quantization_time(start_time[idx], phrase_end_time)
                    else:
                        start, end = start_time[idx], phrase_end_time

                    if pitch_correct:
                        pitch = get_nearest_in_tone_note(note_num+note_range[0])
                    else:
                        pitch = note_num + note_range[0]

                    note = pretty_midi.Note(velocity=127, pitch=pitch, start=start, end=end)

                instrument.notes.append(note)
    instrument.notes.sort(key=lambda note: note.start)

    pm.instruments.append(instrument)
    pm.write(path)


def auto_pitch_correct_midi(ori_path, new_path):
    ori_pm = pretty_midi.PrettyMIDI(ori_path)
    new_pm = pretty_midi.PrettyMIDI()

    for instr in ori_pm.instruments:
        new_instr = pretty_midi.Instrument(instr.program)
        for note in instr.notes:
            new_instr.notes.append(pretty_midi.Note(velocity=127, pitch=get_nearest_in_tone_note(note.pitch),
                                                    start=note.start, end=note.end))
        new_pm.instruments.append(new_instr)
    new_pm.write(new_path)


def quantize_midi(ori_path, new_path):
    ori_pm = pretty_midi.PrettyMIDI(ori_path)
    new_pm = pretty_midi.PrettyMIDI()

    for instr in ori_pm.instruments:
        new_instr = pretty_midi.Instrument(instr.program)
        for note in instr.notes:
            ori_start, ori_end = note.start, note.end
            start, end = get_quantization_time(ori_start, ori_end)
            new_instr.notes.append(pretty_midi.Note(velocity=127, pitch=note.pitch, start=start, end=end))
        new_pm.instruments.append(new_instr)
    new_pm.write(new_path)


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


def get_quantization_time(start, end, bpm=120, shortest_note=(1/16, 1/64)):
    shortest_note_length = (60 / bpm * (shortest_note[0] / (1 / 4)), 60 / bpm * (shortest_note[1] / (1 / 4)))
    start_q = shortest_note_length[0] * int(start / shortest_note_length[0])
    end_q = shortest_note_length[1] * int(math.ceil(end / shortest_note_length[1]))

    return start_q, end_q


def test_nonzeros_generation():
    path = 'E:/jimi_library/unit_riffs/nonzeros/guitar/5f0d0da143063ce12a8f0e6a.npz'
    pm = generate_pm_from_nonzeros(path)
    pm.write('./test_pm.mid')


if __name__ == '__main__':
    test_nonzeros_generation()