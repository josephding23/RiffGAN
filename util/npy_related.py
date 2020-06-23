import numpy as np
import pretty_midi
import math


def generate_nonzeros_from_pm(pm, bpm, length, instr_type):
    assert isinstance(pm, pretty_midi.PrettyMIDI)

    sixty_fourth_length = 60 / bpm / 16
    if instr_type == 'guitar':
        note_range = (36, 96)
        # standard tune: [E2, D6] -> [C2, C7)
    else:
        assert instr_type == 'bass'
        note_range = (24, 72)
        # standard tune: [E1, G4] -> [C1, C5)
    shape = (math.ceil(length), 64, note_range[1] - note_range[0])
    nonzeros = []

    for instr in pm.instruments:
        for note in instr.notes:
            start = int(note.start / sixty_fourth_length)
            end = int(note.end / sixty_fourth_length)
            pitch = note.pitch
            if pitch < note_range[0] or pitch >= note_range[1]:
                continue
            else:
                pitch -= note_range[0]
                for time_raw in range(start, end):
                    segment = int(time_raw / 64)
                    time = time_raw % 64
                    nonzeros.append([segment, time, pitch])
                    if pitch >= 60:
                        print(pitch)
    nonzeros = np.array(nonzeros)
    return nonzeros, shape


def generate_sparse_matrix_from_nonzeros(nonzeros, shape):
    data = np.zeros((shape[0], shape[1], shape[2]), np.float_)
    for nonzero in nonzeros:
        data[nonzero[0], nonzero[1], nonzero[2]] = 1.0
    return data


def plot_data(data, shape):
    import matplotlib.pyplot as plt
    sample_data = data
    dataX = []
    dataY = []
    for time in range(shape[1]):
        for pitch in range(shape[2]):
            if sample_data[time][pitch] > 0.1:
                dataX.append(time)
                dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()

