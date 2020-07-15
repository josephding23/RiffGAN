import numpy as np
import random
from music.custom_elements.rhythm_riff.toolkit import *
from util.npy_related import plot_data


def generate_random_seed(length, instr, root_note='I', pattern='5'):

    if instr == 'guitar':
        note_range = (36, 96)
        # standard tune: [E2, D6] -> [C2, C7)
    else:
        assert instr == 'bass'
        note_range = (24, 72)
        # standard tune: [E1, G4] -> [C1, C5)

    seed = np.zeros(shape=(length, 64, note_range[1]-note_range[0]), dtype=np.float_)

    root_dist = get_relative_distance(root_note)

    total_length = 64 * length

    root_prob1 = int(round(random.uniform(0.125, 0.25) * 16)) / 16
    root_prob2 = int(round(random.uniform(0.125, 0.25) * 16)) / 16
    root_prob3 = int(round(random.uniform(0.125, 0.25) * 16)) / 16
    root_prob4 = int(round(random.uniform(0.125, 0.25) * 16)) / 16
    root_prob = root_prob1 + root_prob2 + root_prob3 + root_prob4
    third_prob = int(round(random.uniform(0, 1 - root_prob) * 16)) / 16
    fifth_prob = int(round(random.uniform(0, 1 - root_prob - third_prob) * 16)) / 16
    fourth_prob = 1 - root_prob - third_prob - fifth_prob

    assert root_prob >= 0 and third_prob >= 0 and fifth_prob >= 0 and fourth_prob >= 0

    chord_pattern = get_chord_pattern(pattern)

    random_dict = {
        'I1': {
            'prob': root_prob1,
        },
        'I2': {
            'prob': root_prob2,
        },
        'I3': {
            'prob': root_prob3,
        },
        'I4': {
            'prob': root_prob4,
        },
        'III': {
            'prob': third_prob
        },
        'V': {
            'prob': fifth_prob
        },
        'IV': {
            'prob': fourth_prob
        }
    }

    order_list = ['III', 'V', 'IV']
    random.shuffle(order_list)
    whole_order_list = ['I1', order_list[0], 'I2', order_list[1], 'I3', order_list[2], 'I4']

    start_time = 0
    for order in whole_order_list:
        if order in ['III', 'IV', 'V']:
            distance = get_relative_distance(order)
        else:
            distance = get_relative_distance(order[0])
        prob = random_dict[order]['prob']
        chord = [root_dist + distance + note for note in chord_pattern]
        chord_length = int(prob * total_length)

        start = start_time
        end = start + chord_length

        start_time = end

        for time in range(start, end):
            if time == length * 64:
                continue
            for note in chord:
                seed[time // 64][time % 64][note] = 1.0

    return seed


if __name__ == '__main__':
    random_seed = np.array([generate_random_seed(1, 'guitar') for _ in range(5)])[:, 0, :, :]
    print(random_seed.shape)
    plot_data(random_seed[0, :, :], (1, 64, 60))
    plot_data(random_seed[3, :, :], (1, 64, 60))
