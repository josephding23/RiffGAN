import numpy as np
import pretty_midi
import math


def generate_nonzeros_from_pm(pm, bpm, length):
    assert isinstance(pm, pretty_midi.PrettyMIDI)

    thirty_second_length = 60 / bpm / 8
    note_range = (24, 108)
    shape = (math.ceil(length / 2), 64, 84)
    nonzeros = []

    for instr in pm.instruments:
        for note in instr.notes:
            start = int(note.start / thirty_second_length)
            end = int(note.end / thirty_second_length)
            pitch = note.pitch
            if pitch < note_range[0] or pitch >= note_range[1]:
                continue
            else:
                pitch -= 24
                for time_raw in range(start, end):
                    segment = int(time_raw / 64)
                    time = time_raw % 64
                    nonzeros.append([segment, time, pitch])
    nonzeros = np.array(nonzeros)
    return nonzeros, shape


def generate_sparse_matrix_from_nonzeros(nonzeros, shape):
    data = np.zeros((shape[0], shape[1], shape[2]), np.float_)
    for nonzero in nonzeros:
        print(nonzero)
        data[nonzero[0], nonzero[1], nonzero[2]] = 1.0
    return data


def plot_data(data):
    import matplotlib.pyplot as plt
    sample_data = data
    dataX = []
    dataY = []
    for time in range(64):
        for pitch in range(84):
            if sample_data[time][pitch] > 0.5:
                dataX.append(time)
                dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()