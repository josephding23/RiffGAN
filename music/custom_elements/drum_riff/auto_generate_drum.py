import pretty_midi
import math
from music.custom_elements.drum_riff.drum_riff import *


def get_measure_num(pm, bpm):
    total_length = pm.get_end_time()
    measure_length = 60 / bpm * 4
    measure_num = math.ceil(total_length / measure_length)
    return measure_num


def get_note_arrange_list(pm, bpm):
    assert isinstance(pm, pretty_midi.PrettyMIDI)
    sixty_fourth_length = 60 / bpm / 16

    measure_num = get_measure_num(pm, bpm)
    note_list = [0 for _ in range(64 * measure_num)]
    for instr in pm.instruments:
        if not instr.is_drum:
            for note in instr.notes:
                start, end = note.start, note.end
                start, end = int(round(start / sixty_fourth_length)), int(round(end / sixty_fourth_length))
                for time in range(start, end):
                    note_list[time] += 1

    return note_list


def get_accent(pm, is_raw=False, bpm=120):
    accent_list = []
    note_arrange_list = get_note_arrange_list(pm, bpm)

    measure_num = get_measure_num(pm, bpm)
    segment_length = measure_num * 64

    for i, note in enumerate(note_arrange_list):
        if i == 0 or i == len(note_arrange_list) - 1:
            continue
        if note_arrange_list[i - 1] < note <= note_arrange_list[i + 1]:
            accent_list.append(i)

    if is_raw:
        return accent_list
    else:
        for i, time in enumerate(accent_list):
            time = int(round(time / 4)) * 4
            if time < segment_length:
                accent_list[i] = time
        return accent_list


def generate_drum_riff(pm, bpm=120):
    accent_list = get_accent(pm)
    total_length = pm.get_end_time()
    measure_length = 60 / bpm * 4

    measure_num = math.ceil(total_length / measure_length)
    sixteenth_num = measure_num * 16

    snare_pattern = ['_' for _ in range(sixteenth_num)]
    hihat_pattern = ['c' for _ in range(sixteenth_num)]
    for accent_time in accent_list:
        snare_pattern[accent_time // 4] = 'x'
        hihat_pattern[accent_time // 4] = 'o'

    snare_str = ''.join(snare_pattern)
    hihat_str = ''.join(hihat_pattern)

    drum_riff = DrumRiff(2)
    drum_riff.set_pattern({
        'snare': snare_str,
        'hi-hat': hihat_str
    })

    instr = drum_riff.get_whole_pm_instr(120)

    return instr


def add_drum_riff_to_pm(path, bpm=120):
    pm = pretty_midi.PrettyMIDI(path)
    drum_instr = generate_drum_riff(pm, bpm)
    pm.instruments.append(drum_instr)
    pm.write(path)


if __name__ == '__main__':
    add_drum_riff_to_pm('../../../data/generated_music/gen1.mid')