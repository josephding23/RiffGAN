from music.pieces.phrase import *

def test_phrase():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    phrase = RhythmPhrase(0, 6, ('C3', 'major'), 120, 26)
    phrase.set_riffs([griff])
    phrase.set_arrangement([(0, 'I'), (0, 'V'), (0, 'IV')])
    phrase.add_riffs_to_pm()
    phrase.save('../../data/custom_element/phrase/test1.mid')


def test_drum_phrase():
    drum_riff = DrumRiff(measure_length=1)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})

    drum_phrase = DrumPhrase(0, length=3, bpm=120)
    drum_phrase.set_riffs([drum_riff])
    drum_phrase.set_arrangement([0, 0, 0])
    drum_phrase.add_riffs_to_pm()
    drum_phrase.save('../../data/custom_element/phrase/test2.mid')