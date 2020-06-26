from riffgan.structure.random_seed import *
import torch
from riffgan.networks.midinet.generator import Generator


def test_generator():
    seed_size = 200
    device = torch.device('cuda')

    noise = torch.randn(1, seed_size, device=device)
    seed = torch.unsqueeze(torch.from_numpy(generate_random_seed(1, 'guitar')), 1).to(
        device=device, dtype=torch.float)

    g = Generator(pitch_range=60).to(device=device)

    fake_data = g(noise, seed, 1)


if __name__ == '__main__':
    test_generator()

