from src.custom_elements.riff import *
from src.pieces.toolkit import *
from src.pieces.phrase import Phrase
import pretty_midi


class Track:
    def __init__(self, name, bpm_list, tonality_list):
        self.name = name

        self.phrases = []

        self.bpm_list = bpm_list
        self.tonality_list = tonality_list

        self.pm = pretty_midi.PrettyMIDI()

    def get_measure_start_time(self, measure):
        start_time = 0
        if len(self.bpm_list) == 1:
            assert self.bpm_list[0]['start_measure'] == 0
            start_time = get_measure_length(self.bpm_list[0]['bpm']) * measure
        elif measure > self.bpm_list[-1]['start_measure']:
            for i in range(len(self.bpm_list)-1):
                start_measure, bpm = self.bpm_list[i]['start_measure'], self.bpm_list[i]['bpm']
                next_measure = self.bpm_list[i+1]['start_measure']

                start_time += (next_measure - start_measure) * get_measure_length(bpm)

            start_time += (measure - self.bpm_list[-1]['start_measure']) * get_measure_length(self.bpm_list[-1]['bpm'])

        else:
            for i in range(len(self.bpm_list)-1):
                start_measure, bpm = self.bpm_list[i]['start_measure'], self.bpm_list[i]['bpm']
                next_measure = self.bpm_list[i + 1]['start_measure']

                if measure <= next_measure:
                    start_time += (measure - start_measure) * get_measure_length(bpm)

                else:
                    start_time += (next_measure - start_measure) * get_measure_length(bpm)
        return start_time

    def add_phrase(self, phrase):
        self.phrases.append(phrase)

    def add_phrases(self, phrases):
        self.phrases += phrases

    def get_phrases_num(self):
        print(len(self.phrases))

    def add_phrases_to_pm(self):
        instr = pretty_midi.Instrument(program=0)

        for phrase in self.phrases:
            instr.program = phrase.instr
            phrase_start = self.get_measure_start_time(phrase.start_measure)
            riff_start = phrase_start
            length_per_measure = get_measure_length(phrase.bpm)

            for arrange in phrase.arrangement:
                riff = phrase.riffs[arrange]

                real_time_stamps = time_stamps_convert(riff.time_stamps, phrase.bpm)
                for i in range(len(real_time_stamps)):
                    start_time, end_time = real_time_stamps[i]
                    start_time += riff_start
                    end_time += riff_start
                    if type(riff.velocity) == int:
                        velocity = riff.velocity
                    else:
                        assert type(riff.velocity) == list
                        velocity = riff.velocity[i]
                    chord = riff.chords[i]

                    for note_dist in chord:
                        note = pretty_midi.Note(velocity=velocity, pitch=note_dist + phrase.root_note,
                                                start=start_time, end=end_time)
                        instr.notes.append(note)

                riff_start += length_per_measure * riff.measure_length

        self.pm.instruments.append(instr)

    def save(self, save_path):
        self.pm.write(save_path)


def test_measure_start():
    track = Track('test', [{'start_measure': 0, 'bpm': 120}, {'start_measure': 2, 'bpm': 60}, {'start_measure': 3, 'bpm': 60}], [])
    print(track.get_measure_start_time(3))


def test_track():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])

    phrase1 = Phrase(0, 6, ('C3', 'major'), 120, 26)
    phrase1.set_riffs([griff])
    phrase1.set_arrangement([0, 0, 0])

    phrase2 = Phrase(7, 6, ('C3', 'major'), 120, 26)
    phrase2.set_riffs([griff])
    phrase2.set_arrangement([0, 0, 0])

    track = Track(name='test',
                  bpm_list=[{'start_measure': 0, 'bpm': 120}],
                  tonality_list=[{}])
    track.add_phrases([phrase1, phrase2])
    track.add_phrases_to_pm()
    track.save('../../data/custom_element/track/test1.mid')


if __name__ == '__main__':
    test_track()