from src.pieces.song import *


def test_phrase():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    phrase = Phrase(0, 6, ('C3', 'major'), 120, 26)
    phrase.set_riffs([griff])
    phrase.set_arrangement([(0, 'I'), (0, 'V'), (0, 'IV')])
    phrase.add_riffs_to_pm()
    phrase.save('../../data/custom_element/phrase/test1.mid')


def test_measure_start():
    track = Track('test',
                  [(0, 120), (1, 60), (3, 60)],
                  [])
    print(track.get_measure_start_time(3))


def test_track():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])

    phrase1 = Phrase(0, 6, ('C3', 'major'), 120, 26)
    phrase1.set_riffs([griff])
    phrase1.set_arrangement([(0, 'I'), (0, 'III'), (0, 'V')])

    phrase2 = Phrase(7, 6, ('C3', 'major'), 120, 26)
    phrase2.set_riffs([griff])
    phrase2.set_arrangement([(0, 'I'), (0, 'III'), (0, 'V')])

    track = Track(name='test',
                  bpm_list=[(0, 120)],
                  tonality_list=[{}])
    track.set_phrases([phrase1, phrase2])
    track.add_phrases_to_pm()
    track.save('../../data/custom_element/track/test1.mid')


def test_song():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('IV', ''), ('V', ''), ('IV', ''), ('III', ''), ('I', '')],
                       time_stamps=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2,
                                    1/2, 1, 1/2, 1/2, 1,
                                    1/8, 1/8, 1/4, 1/2, 1/2])
    briff = generate_briff_from_griff(griff)

    phrase11 = Phrase(0, 6, ('G2', 'major'), 120, 29)
    phrase11.set_riffs([griff])
    phrase11.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    phrase12 = Phrase(7, 6, ('G2', 'major'), 120, 29)
    phrase12.set_riffs([griff])
    phrase12.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    track_guitar = Track(name='guitar',
                         bpm_list=[(0, 120)],
                         tonality_list=[{}])
    track_guitar.set_phrases([phrase11, phrase12])
    track_guitar.add_phrases_to_pm()

    phrase21 = Phrase(0, 6, ('G1', 'major'), 120, 33)
    phrase21.set_riffs([briff])
    phrase21.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    phrase22 = Phrase(7, 6, ('G1', 'major'), 120, 33)
    phrase22.set_riffs([briff])
    phrase22.set_arrangement([(0, 'I'), (0, 'V'), (0, 'III')])

    track_bass = Track(name='bass',
                       bpm_list=[(0, 120)],
                       tonality_list=[{}])
    track_bass.set_phrases([phrase21, phrase22])

    song = Song('test_song')
    song.set_tracks([track_guitar, track_bass])
    song.add_tracks_to_pm()
    song.save('../../data/custom_element/song/test1.mid')
