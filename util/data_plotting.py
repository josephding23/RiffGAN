import pretty_midi
from util.npy_related import generate_nonzeros_from_pm
import matplotlib.pyplot as plt
import numpy as np


def plot_midi_file(path, length, instr_type, bpm=120, save_image=False, save_path=None):
    pm = pretty_midi.PrettyMIDI(path)
    nonzeros, shape = generate_nonzeros_from_pm(pm, bpm, length, instr_type)
    plot_nonzeros(nonzeros, shape, save_image=save_image, save_path=save_path)


def plot_data(data):
    import matplotlib.pyplot as plt
    shape = data.shape
    sample_data = data
    dataX = []
    dataY = []

    for i in range(shape[0]):
        for time in range(shape[1]):
            for pitch in range(shape[2]):
                if sample_data[i][time][pitch] > 0.1:
                    dataX.append(time + i * shape[1])
                    dataY.append(pitch)
    plt.scatter(x=dataX, y=dataY)
    plt.show()


def plot_nonzeros(nonzeros, shape, save_image=False, save_path=None):
    import matplotlib.pyplot as plt
    dataX = []
    dataY = []

    print(shape)

    if shape[2] == 60:
        pitch_labels = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    else:
        pitch_labels = ['C1', 'C2', 'C3', 'C4', 'C5']

    time_ticks = [time * 16 for time in range(shape[0] * shape[1] // 16 + 1)]
    pitch_ticks = [pitch * 12 for pitch in range(shape[2] // 12 + 1)]

    print(time_ticks, pitch_ticks)

    fig = plt.figure()
    fig = plt.figure(figsize=(7.2, 2.7), dpi=100)

    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])

    for nonzero in nonzeros:
        dataX.append(nonzero[0] * 64 + nonzero[1])
        dataY.append(nonzero[2])
    ax.scatter(x=dataX, y=dataY)

    ax.set_xlim(0, shape[0] * shape[1])
    ax.set_ylim(0, shape[2])
    ax.set_xticks(time_ticks, time_ticks)
    ax.set_yticks(pitch_ticks, pitch_labels)

    if save_image:
        plt.savefig(save_path)
    plt.show()


def plot_midi_and_save(midi_path, save_path, length, instr_type, bpm=120):
    fig = plt.figure()
    plot_midi_file(midi_path, length, instr_type, bpm)
    fig.savefig(save_path)


def test_plot():
    nonzeros = []
    plot_nonzeros(nonzeros, shape=[2, 64, 60])


if __name__ == '__main__':
    test_plot()
