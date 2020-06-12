from music.pieces.phrase import *


def test_rhythm_phrase():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    rhythm_phrase = RhythmPhrase(6, ['C3', 'major'], 120, 26, "guitar")
    rhythm_phrase.set_riffs([griff])
    rhythm_phrase.set_arrangement([(0, 'I'), (0, 'V'), (0, 'IV')])
    rhythm_phrase.add_riffs_to_pm()
    rhythm_phrase.save_midi('test1')
    # phrase.play_it()
    rhythm_phrase.save_json("test1")


def test_drum_phrase():
    drum_riff = DrumRiff(measure_length=1)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})

    drum_phrase = DrumPhrase(length=3, bpm=120)
    drum_phrase.set_riffs([drum_riff])
    drum_phrase.set_arrangement([0, 0, 0])
    drum_phrase.add_riffs_to_pm()
    drum_phrase.save_midi('drum1')
    drum_phrase.save_json('drum1')


def drum_phrase_from_json():
    drum_phrase = create_drum_phrase_from_json('D:/PycharmProjects/RiffGAN/data/pieces/phrases/json/drum1.json')
    drum_phrase.add_riffs_to_pm()
    drum_phrase.save_midi('drum_from_midi')
    drum_phrase.play_it()


def rhythm_phrase_from_json():
    rhythm_phrase = create_rhythm_phrase_from_json('D:/PycharmProjects/RiffGAN/data/pieces/phrases/json/test1.json')
    rhythm_phrase.add_riffs_to_pm()
    rhythm_phrase.save_midi('rhythm_from_midi')
    rhythm_phrase.play_it()


if __name__ == '__main__':
    rhythm_phrase_from_json()