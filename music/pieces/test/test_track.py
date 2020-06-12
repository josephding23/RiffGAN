from music.pieces.track import *


def test_track():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])

    phrase1 = RhythmPhrase(0, 6, ('C3', 'major'), 120, 26)
    phrase1.set_riffs([griff])
    phrase1.set_arrangement([(0, 'I'), (0, 'III'), (0, 'V')])

    phrase2 = RhythmPhrase(7, 6, ('C3', 'major'), 120, 26)
    phrase2.set_riffs([griff])
    phrase2.set_arrangement([(0, 'I'), (0, 'III'), (0, 'V')])

    track = Track(name='test',
                  bpm_list=[(0, 120)],
                  tonality_list=[{}])
    track.set_phrases([phrase1, phrase2])
    track.add_phrases_to_pm()
    track.save('../../data/custom_element/track/test1.mid')


def test_drum_track():
    drum_riff = DrumRiff(measure_length=1)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})

    drum_phrase1 = DrumPhrase(0, length=3, bpm=120)
    drum_phrase1.set_riffs([drum_riff])
    drum_phrase1.set_arrangement([0, 0, 0])

    drum_phrase2 = DrumPhrase(4, length=3, bpm=120)
    drum_phrase2.set_riffs([drum_riff])
    drum_phrase2.set_arrangement([0, 0, 0])

    track = Track('drum_test', [(0, 120)], [], True)
    track.set_phrases([drum_phrase1, drum_phrase2])
    track.add_phrases_to_pm()
    track.save('../../../data/custom_element/track/test2.mid')


def test_track_json_not_drum():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])

    phrase1 = RhythmPhrase(6, ('C3', 'major'), 120, 26, 'guitar')
    phrase1.set_riffs([griff])
    phrase1.set_arrangement([(0, 'I'), (0, 'III'), (0, 'V')])

    track = Track(name='test_no_drum',
                  bpm_list=[(0, 120)],
                  tonality_list=[{}])
    track.set_phrases([phrase1])
    track.set_arrangement([[0, 0], [0, 6]])

    track.save_json()


def test_track_json_drum():
    drum_riff = DrumRiff(measure_length=1)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})

    drum_phrase1 = DrumPhrase(length=3, bpm=120)
    drum_phrase1.set_riffs([drum_riff])
    drum_phrase1.set_arrangement([0, 0, 0])

    track = Track('drum_test', [(0, 120)], [], True)
    track.set_phrases([drum_phrase1])
    track.set_arrangement([[0, 0], [0, 6]])
    track.add_phrases_to_pm()

    track.save_json()


def track_from_json():
    drum_path = 'D:/PycharmProjects/RiffGAN/data/pieces/tracks/json/drum_test.json'
    not_drum_path = 'D:/PycharmProjects/RiffGAN/data/pieces/tracks/json/test_no_drum.json'

    track = create_track_from_json(not_drum_path)
    track.add_phrases_to_pm()
    track.save_midi()
    track.play_it()


if __name__ == '__main__':
    track_from_json()