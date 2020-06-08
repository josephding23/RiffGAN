from music.pieces.song import *


def test_song():
    # Guitar track
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('IV', ''), ('V', ''), ('IV', ''), ('III', ''), ('I', '')],
                       time_stamps=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2,
                                    1/2, 1, 1/2, 1/2, 1,
                                    1/8, 1/8, 1/4, 1/2, 1/2])

    phrase11 = RhythmPhrase(0, 6, ('G2', 'major'), 120, 29)
    phrase11.set_riffs([griff])
    phrase11.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    phrase12 = RhythmPhrase(7, 6, ('G2', 'major'), 120, 29)
    phrase12.set_riffs([griff])
    phrase12.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    track_guitar = Track(name='guitar',
                         bpm_list=[(0, 120)],
                         tonality_list=[{}])
    track_guitar.set_phrases([phrase11, phrase12])
    track_guitar.add_phrases_to_pm()

    # Bass track
    briff = generate_briff_from_griff(griff)

    phrase21 = RhythmPhrase(0, 6, ('G1', 'major'), 120, 33)
    phrase21.set_riffs([briff])
    phrase21.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    phrase22 = RhythmPhrase(7, 6, ('G1', 'major'), 120, 33)
    phrase22.set_riffs([briff])
    phrase22.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    track_bass = Track(name='bass',
                       bpm_list=[(0, 120)],
                       tonality_list=[{}])
    track_bass.set_phrases([phrase21, phrase22])

    # Track Drum
    drum_riff = DrumRiff(measure_length=1)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})

    phrase31 = DrumPhrase(0, length=6, bpm=120)
    phrase31.set_riffs([drum_riff])
    phrase31.set_arrangement([0, 0, 0, 0, 0, 0])

    phrase32 = DrumPhrase(7, length=6, bpm=120)
    phrase32.set_riffs([drum_riff])
    phrase32.set_arrangement([0, 0, 0, 0, 0, 0])

    track_drum = Track(name='drum',
                       bpm_list=[(0, 120)],
                       tonality_list=[], is_drum=True)
    track_drum.set_phrases([phrase31, phrase32])

    song = Song('test_song')
    song.set_tracks([track_guitar, track_bass, track_drum])
    song.add_tracks_to_pm()
    song.save('../../../data/custom_element/song/test2.mid')


if __name__ == '__main__':
    test_song()
