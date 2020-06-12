from util.npy_related import *
from music.db_fragments.riff import UnitGuitarRiff
from dataset.grunge_librarypy import *

def generate_unit_riff_nonzeros():
    unit_guitar_riff_dir = 'E:/unit_riffs/transposed/guitar/'
    unit_bass_riff_dir = 'E:/unit_riffs/transposed/bass/'

    unit_guitar_nonzeros_dir = 'E:/unit_riffs/nonzeros/guitar/'
    unit_bass_nonzeros_dir = 'E:/unit_riffs/nonzeros/bass/'

    unit_guitar_table = get_unit_guitar_riff_table()
    unit_bass_table = get_unit_bass_riff_table()

    for ugriff in unit_guitar_table.find():
        riff_path = unit_guitar_riff_dir + ugriff['idStr'] + '.mid'
        nonzeros_path = unit_guitar_nonzeros_dir + ugriff['idStr'] + '.npz'
        unit_guitar_riff = UnitGuitarRiff(riff_path)
        unit_guitar_riff.save_nonzeros(nonzeros_path)

    for ubriff in unit_bass_table.find():
        riff_path = unit_bass_riff_dir + ubriff['idStr'] + '.mid'
        nonzeros_path = unit_bass_nonzeros_dir + ubriff['idStr'] + '.npz'
        unit_bass_riff = UnitGuitarRiff(riff_path)
        unit_bass_riff.save_nonzeros(nonzeros_path)


def create_nonzeros():
    unit_guitar_nonzeros_dir = 'E:/grunge_library/unit_riffs/nonzeros/guitar/'
    unit_bass_nonzeros_dir = 'E:/grunge_library/unit_riffs/nonzeros/bass/'

    guitar_nonzeros_path = 'E:/grunge_library/dataset/guitar_unit_riff.npz'
    bass_nonzeros_path = 'E:/grunge_library/dataset/bass_unit_riff.npz'

    unit_guitar_table = get_unit_guitar_riff_table()
    unit_bass_table = get_unit_bass_riff_table()

    ugriff_num = unit_guitar_table.count()
    ubriff_num = unit_bass_table.count()

    unit_guitar_nonzeros = []
    unit_bass_nonzeros = []

    ugriff_no = 0
    for ugriff in unit_guitar_table.find():
        nonzeros_path = unit_guitar_nonzeros_dir + ugriff['idStr'] + '.npz'
        unit_nonzeros = np.load(nonzeros_path)['nonzeros']
        for nonzero in unit_nonzeros:
            unit_guitar_nonzeros.append([ugriff_no, nonzero[1], nonzero[2]])
        ugriff_no += 1

    ubriff_no = 0
    for ubriff in unit_bass_table.find():
        nonzeros_path = unit_bass_nonzeros_dir + ubriff['idStr'] + '.npz'
        nonzeros = np.load(nonzeros_path)['nonzeros']
        for nonzero in nonzeros:
            unit_bass_nonzeros.append([ubriff_no, nonzero[1], nonzero[2]])
        ubriff_no += 1

    np.savez_compressed(guitar_nonzeros_path, nonzeros=unit_guitar_nonzeros, shape=(ugriff_num, 64, 84))
    np.savez_compressed(bass_nonzeros_path, nonzeros=unit_bass_nonzeros, shape=(ubriff_num, 64, 84))


def generate_from_nonzeros(instr):
    assert instr in ['guitar', 'bass']

    dataset_dict = {
        'guitar': 'E:/grunge_library/dataset/guitar_unit_riff.npz',
        'bass': 'E:/grunge_library/dataset/bass_unit_riff.npz'
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

        plot_data(data[2, ...])
    return data


if __name__ == '__main__':
    create_nonzeros()