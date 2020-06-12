from music.custom_elements.riff import *
from music.custom_elements.drum_riff import *


def test_riff():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1/2, 1/2, 1/2])
    griff.add_notes_to_pm(root_note_name='G2', bpm=120, instr=27)
    griff.save_midi('test1')
    griff.save_json('test1')
    griff.play_it()


def griff_from_json():
    json_path = 'D:/riff_gan/data/custom_element/guitar_riff/json/test1.json'
    griff = create_griff_from_json(json_path)
    griff.add_notes_to_pm(root_note_name='G2', bpm=120, instr=27)
    griff.save_midi('test_json')
    griff.play_it()


def test_plot():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    griff.add_notes_to_pm(root_note_name='C3', bpm=120, instr=27)

    nonzeros, shape = generate_nonzeros_from_pm(griff.pm, 120, 2)
    data = generate_sparse_matrix_from_nonzeros(nonzeros, shape)
    plot_data(data[0])


def test_simple_briff():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    briff_simple = generate_briff_from_griff(griff)
    briff_simple.add_notes_to_pm(root_note_name='G1', bpm=120, instr=33)
    briff_simple.save_midi('test1')
    briff_simple.play_it()


def test_drum_riff():
    drum_riff = DrumRiff(measure_length=2)
    drum_riff.set_pattern({'hi-hat': 'ccccoccococco_cc', 'snare': '____x__x_x__x___'})
    drum_riff.add_all_patterns_to_pm(120)
    drum_riff.save_json('drum1')


def driff_from_json():
    json_path = 'D:/riff_gan/data/custom_element/drum_riff/json/drum1.json'
    driff = create_driff_from_json(json_path)
    driff.add_all_patterns_to_pm(120)
    driff.save_midi('drum_from_json')
    driff.play_it()


def test_eq():
    griff1 = GuitarRiff(measure_length=2,
                        degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                        time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    griff2 = GuitarRiff(measure_length=2,
                        degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                        time_stamps=[1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 1, 1 / 2, 1 / 2, 1 / 2])
    print(griff1 == griff2)


if __name__ == '__main__':
    test_eq()
