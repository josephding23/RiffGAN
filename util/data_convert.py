import pretty_midi
import numpy as np


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


def save_midis(bars, path):
    from util.npy_related import plot_data
    pm = pretty_midi.PrettyMIDI()

    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 24)),
                                  bars,
                                  np.zeros((bars.shape[0], bars.shape[1], bars.shape[2], 20))),
                                 axis=3)
    padded_bars = padded_bars.reshape((-1, padded_bars.shape[1], padded_bars.shape[2], padded_bars.shape[3]))
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[1]):
        padded_bars_list.append(padded_bars[:, ch_idx, :, :].reshape(padded_bars.shape[0],
                                                                     padded_bars.shape[2],
                                                                     padded_bars.shape[3]))

    pianoroll = padded_bars_list[0]
    pianoroll = pianoroll.reshape((pianoroll.shape[0] * pianoroll.shape[1], pianoroll.shape[2]))
    pianoroll_diff = np.concatenate((np.zeros((1, 128), dtype=int), pianoroll, np.zeros((1, 128), dtype=int)))
    pianoroll_search = np.diff(pianoroll_diff.astype(int), axis=0)

    instrument = pretty_midi.Instrument(program=0, is_drum=False, name='Instr')

    tempo = 120
    beat_resolution = 16

    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 16
    phrase_end_time = 60.0 / tempo * 16 * pianoroll.shape[0]

    for note_num in range(128):
        start_idx = (pianoroll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))

        end_idx = (pianoroll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))

        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        t.append(j)
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
                note = pretty_midi.Note(velocity=127, pitch=note_num, start=start_time[idx], end=end_time[idx])
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    note = pretty_midi.Note(velocity=127, pitch=note_num, start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    note = pretty_midi.Note(velocity=127, pitch=note_num, start=start_time[idx],
                                            end=phrase_end_time)
                instrument.notes.append(note)
    instrument.notes.sort(key=lambda note: note.start)

    pm.instruments.append(instrument)
    pm.write(path)
