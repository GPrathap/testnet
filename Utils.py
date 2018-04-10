
import numpy as np

from grid_layout import create_mine_grid


def get_init_vector(length, batch_size):
    noise1 = create_mine_grid(1, length, batch_size, length - 1, None, True, True)
    noise = np.transpose(noise1)
    return np.array(noise, dtype=np.float32)