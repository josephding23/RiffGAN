from util.npy_related import *
from music.db_fragments.riff import UnitGuitarRiff, UnitBassRiff
from dataset.grunge_library import *


def generate_from_nonzeros(source, instr):
    assert instr in ['guitar', 'bass']

    if source is 'grunge_library':
        dataset_dict = {
            'guitar': 'E:/riff_data/grunge_library/data/guitar_unit_riff.npz',
            'bass': 'E:/riff_data/grunge_library/data/bass_unit_riff.npz'
        }
    else:
        assert source is 'jimi_library'
        dataset_dict = {
            'guitar': 'E:/riff_data/jimi_library/data/guitar_unit_riff.npz',
            'bass': 'E:/riff_data/jimi_library/data/bass_unit_riff.npz'
        }

    path = dataset_dict[instr]

    with np.load(path) as npz_file:
        nonzeros = npz_file['nonzeros']
        shape = npz_file['shape']

        data = np.zeros(shape, np.float_)
        print(data.shape)
        for nonzero in nonzeros:
            # print(nonzero)
            data[(int(nonzero[0]), int(nonzero[1]), int(nonzero[2]))] = 1.0

        # plot_data(data[2, ...])
    return data


if __name__ == '__main__':
    dataset = generate_from_nonzeros('guitar')
    plot_data(dataset[0, :, :], (1, 64, 60))