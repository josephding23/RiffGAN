import pretty_midi
from operator import itemgetter


def merge_short_notes(ori_path, new_path, instr_type, bpm=120):
    sixty_fourth_length = 60 / bpm / 16 * 1.1

    sixteenth_length = 60 / bpm / 4

    if instr_type == 'guitar':
        note_range = (36, 96)
        instr_num = 29
        # standard tune: [E2, D6] -> [C2, C7)
    else:
        assert instr_type == 'bass'
        note_range = (24, 72)
        instr_num = 33
        # standard tune: [E1, G4] -> [C1, C5)

    pitch_list = [[] for _ in range(note_range[1] - note_range[0])]

    pm = pretty_midi.PrettyMIDI(ori_path)
    for instr in pm.instruments:
        if instr.is_drum:
            break
        for note in instr.notes:
            pitch = note.pitch
            start = note.start
            end = note.end

            _pitch = pitch-note_range[0]
            pitch_list[_pitch].append({'start': start, 'end': end, 'ignore_first': False})

    pitch_list = [sorted(pitch_list[_pitch], key=itemgetter('start')) for _pitch in range(note_range[1] - note_range[0])]

    new_pitch_list = [[] for _ in range(note_range[1] - note_range[0])]

    for _pitch, time_info_list in enumerate(pitch_list):
        for i, time_info in enumerate(time_info_list):

            if time_info['ignore_first']:
                continue

            start, end = time_info['start'], time_info['end']
            for j in range(1, len(time_info_list) - i):
                current_time_info = time_info_list[i+j]
                current_start, current_end = current_time_info['start'], current_time_info['end']

                if current_start - end <= sixty_fourth_length:
                    end = current_end
                    time_info_list[i+j]['ignore_first'] = True
                else:
                    new_pitch_list[_pitch].append({'start': start, 'end': end})
                    break
            new_pitch_list[_pitch].append({'start': start, 'end': end})

    instr = pretty_midi.Instrument(program=instr_num)
    for _pitch, time_info_list in enumerate(new_pitch_list):
        pitch = _pitch + note_range[0]
        for time_info in time_info_list:
            start, end = time_info['start'], time_info['end']
            if end - start < sixty_fourth_length:
                continue
            start = round(start / sixteenth_length) * sixteenth_length
            end = round(end / sixteenth_length) * sixteenth_length

            instr.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))

    pm = pretty_midi.PrettyMIDI()
    pm.instruments.append(instr)

    # new_path = path[:-4] + '_merged_.mid'
    pm.write(new_path)


def test_merge():
    path = '../data/generated_music/gen2.mid'
    merge_short_notes(path, 'guitar')


if __name__ == '__main__':
    test_merge()