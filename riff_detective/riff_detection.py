from dataset.free_midi_library import *
import pretty_midi


def count_bar_length(bpm, metre):
    numerator, denominator = metre
    quarter_note_length = 60 / bpm
    denom_note_length = quarter_note_length * 4 / int(denominator)

    bar_length = denom_note_length * numerator
    return bar_length


def detect_riff():
    midi_table = get_midi_table()
    for midi in midi_table.find({'Genre': {'$ne': 'pop'}}):
        path = 'E:/free_midi_library/raw_midi/' + midi['Genre'] + '/' + midi['md5'] + '.mid'
        pm = pretty_midi.PrettyMIDI(path)
        time_info_list = midi['TimeInfoList']

        bpm = time_info_list[0]['tempo']
        # numerator, denominator = midi['MetaInfo']['time_signature'][0]['numerator'], midi['MetaInfo']['time_signature'][0]['denominator']
        for i, time_info in enumerate(time_info_list):
            start_time = time_info['time']
            if i == len(time_info_list) - 1:
                end_time = pm.get_end_time()
            else:
                end_time = time_info_list[i+1]['time']

            length = end_time - start_time

            if length < 0.1:
                print(path)
                break

            if 'tempo' in time_info.keys():
                bpm = time_info['tempo']

            if 'numerator' in time_info.keys():
                numerator, denominator = time_info['numerator'], time_info['denominator']

            # bar_length = count_bar_length(bpm, (numerator, denominator))

            # print(start_time, end_time)


if __name__ == '__main__':
    detect_riff()