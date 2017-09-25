import numpy as np
import itertools


def make_lr_batch_size_grid(max_lr=1.0, lr_step=0.1, min_lr_factor=10,
                            min_batch_size=4, batch_size_ste=2, max_batch_size_factor=8):

    possible_learning_rates = np.asarray([max_lr * lr_step ** (n - 1) for n in range(1, min_lr_factor + 1)])

    possible_batch_sizes = np.asarray([min_batch_size * batch_size_ste ** (n - 1) for n in range(1, max_batch_size_factor + 1)])

    grid = list(itertools.product(possible_learning_rates, possible_batch_sizes))

    return grid
