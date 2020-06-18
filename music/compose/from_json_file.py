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
            griffs_list.append(parse_griff_json(griff))

        for briff in briffs_info:
            if 'from_griff' in briff.keys():
                from_griff = briff['from_griff']
                briffs_list.append(generate_briff_from_griff(griffs_list[from_griff]))
            else:
                briffs_list.append(parse_briff_json(briff))

        for driff in driffs_info:
            driffs_list.append(parse_driff_json(driff))

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

            rhythm_phrases_list.append(phrase)

        for phr in drum_phrases_info:
            start_measure = phr['start']
            length = phr['length']
            bpm = phr['bpm']

            phr_riffs = [driffs_list[num] for num in phr['riffs']]
            phr_arrangement = phr['arrangement']

            phrase = DrumPhrase(start_measure=start_measure, length=length, bpm=bpm)
            phrase.set_riffs(phr_riffs)
            phrase.set_arrangement(phr_arrangement)

            drum_phrases_list.append(phrase)

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

        song = Song(name=song_name)
        song.set_tracks(tracks_list)
        return song


def test_parse_json():
    json_path = '/PycharmProjects/RiffGAN/database/jsons/test1.json'
    song = parse_song_from_json(json_path)
    song.add_tracks_to_pm()
    song.save_midi()
    song.play_it()


if __name__ == '__main__':
    test_parse_json()