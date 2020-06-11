import json
from music.pieces.song import *


def parse_song_from_json(path):
    with open(path, 'r') as f:
        song_info = json.loads(f.read())

        song_name = song_info['name']

        griffs_info = song_info['griffs']
        briffs_info = song_info['briffs']
        driffs_info = song_info['driffs']
        rhythm_phrases_info = song_info['rhythm_phrases']
        drum_phrases_info = song_info['drum_phrases']
        tracks_info = song_info['tracks']

        griffs_list = []
        briffs_list = []
        driffs_list = []
        rhythm_phrases_list = []
        drum_phrases_list = []
        tracks_list = []

        assert isinstance(griffs_info, list)
        assert isinstance(briffs_info, list)
        assert isinstance(driffs_info, list)
        assert isinstance(rhythm_phrases_info, list)
        assert isinstance(drum_phrases_info, list)
        assert isinstance(tracks_info, list)

        '''
        griffs_num = len(griffs_info)
        briffs_num = len(griffs_info)
        driffs_num = len(driffs_info)
        rhythm_phrases_num = len(rhythm_phrases_info)
        drum_phrases_num = len(drum_phrases_info)
        tracks_num = len(tracks_info)
        '''

        for griff in griffs_info:
            measure_length = griff['length']
            degrees_and_types = griff['degrees_and_types']
            time_stamps = griff['time_stamps']

            griffs_list.append(GuitarRiff(measure_length=measure_length,
                                          degrees_and_types=degrees_and_types,
                                          time_stamps=time_stamps)
                               )

        for briff in briffs_info:
            from_griff = briff['from_griff']
            if from_griff == -1:
                measure_length = briff['measure_length']
                degrees_and_types = briff['degrees_and_types']
                time_stamps = briff['time_stamps']

                briffs_list.append(BassRiff(measure_length=measure_length,
                                            degrees_and_types=degrees_and_types,
                                            time_stamps=time_stamps)
                                   )
            else:
                briffs_list.append(generate_briff_from_griff(griffs_list[from_griff]))

        for driff in driffs_info:
            measure_length = driff['measure_length']
            pattern = driff['pattern']

            drum_riff = DrumRiff(measure_length=measure_length)
            drum_riff.set_pattern(pattern)
            driffs_list.append(drum_riff)

        for phr in rhythm_phrases_info:
            start_measure = phr['start']
            length = phr['length']
            tonality = phr['tonality']
            bpm = phr['bpm']

            instr = phr['instr']
            instr_type = phr['instr_type']

            if instr_type == "guitar":
                phr_riffs = [griffs_list[riff_num] for riff_num in phr['riffs']]
            else:
                assert instr_type == 'bass'
                phr_riffs = [briffs_list[riff_num] for riff_num in phr['riffs']]

            phr_arrangement = phr['arrangement']

            phrase = RhythmPhrase(start_measure=start_measure, length=length,
                                  tonality=tonality, bpm=bpm, instr=instr)
            phrase.set_riffs(phr_riffs)
            phrase.set_arrangement(phr_arrangement)

        for phr in driffs_info:
            start_measure = phr['start']
            length = phr['length']
            bpm = phr['bpm']

            phr_riffs = phr['riffs']
            phr_arrangement = phr['arrangement']

            phrase = DrumPhrase(start_measure=start_measure, length=length, bpm=bpm)
            phrase.set_riffs(phr_riffs)
            phrase.set_arrangement(phr_arrangement)

        for tr in tracks_info:
            name = tr['name']
            bpm_list = tr['bpm_list']
            tonality_list = tr['tonality_list']
            is_drum = tr['is_drum']
            tr_phrases = tr['phrases']

            track = Track(name=name, bpm_list=bpm_list, tonality_list=tonality_list, is_drum=is_drum)
            if is_drum:
                track.set_phrases([drum_phrases_list[num] for num in tr_phrases])
            else:
                track.set_phrases([rhythm_phrases_list[num] for num in tr_phrases])

            tracks_list.append(track)

        song = Song([song_name])
        return song


def test_parse_json():
    json_path = '../../data/jsons/test1.json'
    song = parse_song_from_json(json_path)
    song.add_tracks_to_pm()
    song.save()


if __name__ == '__main__':
    test_parse_json()