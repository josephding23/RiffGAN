import numpy as np
import random
from music.custom_elements.toolkit import *
from util.npy_related import plot_data


def generate_random_seed(length, instr='guitar', pattern='5'):
    random_seed = np.zeros(shape=(length, 64, 84), dtype=np.float_)

    for i in range(length):
        root_prob = round(random.uniform(0.3, 0.6), 2)
        third_prob = round(random.uniform(0.2, 0.8 - root_prob), 2)
        fifth_prob = round(random.uniform(0.2, 0.9 - root_prob - third_prob), 2)
        second_prob = round(1 - (root_prob + third_prob + fifth_prob), 2)

        assert root_prob >= 0 and third_prob >= 0 and fifth_prob >= 0 and second_prob >= 0

        chord_pattern = get_chord_pattern(pattern)

        random_dict = {
            'root': {
                'prob': root_prob,
            },
            'third': {
                'prob': third_prob
            },
            'fifth': {
                'prob': fifth_prob
            },
            'second': {
                'prob': second_prob
            }
        }

        if instr == 'guitar':
            random_dict['root']['note'] = 'E2'
            random_dict['third']['note'] = '+G2'
            random_dict['fifth']['note'] = 'B2'
            random_dict['second']['note'] = '+F2'
        else:
            assert instr == 'bass'
            random_dict['root']['note'] = 'E1'
            random_dict['third']['note'] = '+G1'
            random_dict['fifth']['note'] = 'B1'
            random_dict['second']['note'] = '+F1'

        order_list = ['root', 'third', 'fifth', 'second']
        random.shuffle(order_list)

        start_time = 0
        total_length = 64
        for order in order_list:
            root_note = note_name_to_num(random_dict[order]['note'])
            prob = random_dict[order]['prob']
            chord = [note + root_note for note in chord_pattern]

            start = start_time
            end = start + int(prob * total_length)

            start_time = end

            for time in range(start, end):
                for note in chord:
                    random_seed[i][time][note] = 1.0

    return random_seed


if __name__ == '__main__':
    random_seed = generate_random_seed(2)
    plot_data(random_seed[0, :, :])
    plot_data(random_seed[1, :, :])