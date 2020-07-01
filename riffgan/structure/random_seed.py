import numpy as np
import random
from music.custom_elements.riff.toolkit import *
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

    for i in range(length):
        root_prob = round(random.uniform(0.6, 0.8), 1)
        third_prob = round(random.uniform(0.05, 0.9 - root_prob), 1)
        fifth_prob = round(random.uniform(0.05, 0.9 - root_prob - third_prob), 1)
        fourth_prob = round(1 - (root_prob + third_prob + fifth_prob), 1)

        assert root_prob >= 0 and third_prob >= 0 and fifth_prob >= 0 and fourth_prob >= 0

        chord_pattern = get_chord_pattern(pattern)

        random_dict = {
            'I': {
                'prob': root_prob,
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

        order_list = ['I', 'III', 'V', 'IV']
        random.shuffle(order_list)

        start_time = 0
        total_length = 64
        for order in order_list:
            distance = get_relative_distance(order)
            prob = random_dict[order]['prob']
            chord = [root_dist + distance + note for note in chord_pattern]

            start = start_time
            end = start + int(prob * total_length)

            start_time = end

            for time in range(start, end):
                for note in chord:
                    seed[i][time][note] = 1.0

    return seed


if __name__ == '__main__':
    random_seed = generate_random_seed(2, 'guitar')
    plot_data(random_seed[0, :, :], (1, 64, 60))
    plot_data(random_seed[1, :, :], (1, 64, 60))