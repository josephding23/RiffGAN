from src.custom_elements.riff import *


def test_riff():
    griff = GuitarRiff(measure_length=2,
                       degrees_and_types=[('I', '5'), ('I', '5'), ('II', '5'), ('V', '5'), ('III', '5'), ('I', '5'),
                                          ('III', '5'), ('VI', '5'), ('V', '5'), ('III', '5'), ('I', '5')],
                       time_stamps=[1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1/2, 1/2, 1/2])
    griff.add_notes_to_pm(root_note_name='G2', bpm=120, instr=27)
    griff.save('../../data/custom_element/guitar_riff/test1.mid')


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
    briff_simple.save('../../data/custom_element/bass_riff/test1.mid')
