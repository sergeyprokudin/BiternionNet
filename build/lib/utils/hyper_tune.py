import numpy as np
import itertools


def make_lr_batch_size_grid(max_lr=1.0, lr_step=0.1, min_lr_factor=10,
                            min_batch_size=4, batch_size_ste=2, max_batch_size_factor=8):

    possible_learning_rates = np.asarray([max_lr * lr_step ** (n - 1) for n in range(1, min_lr_factor + 1)])

    possible_batch_sizes = np.asarray([min_batch_size * batch_size_ste ** (n - 1) for n in range(1, max_batch_size_factor + 1)])

    grid = list(itertools.product(possible_learning_rates, possible_batch_sizes))

    return grid


def sample_exp_float(n_samples, base, min_factor, max_factor):

    samples = np.power(np.ones(n_samples)*base, np.random.rand(n_samples)*(max_factor-min_factor) + min_factor)

    return samples


def sample_exp_int(n_samples, base, min_factor, max_factor):

    samples = sample_exp_float(n_samples, base, min_factor, max_factor)

    return samples.astype('int')


def sample_batch_size(n_samples, min_batch_factor=1, max_batch_size_factor=10):

    samples = sample_exp_int(n_samples, base=2, min_factor=min_batch_factor, max_factor=max_batch_size_factor)

    return samples


def sample_learning_rates(n_samples, min_lr_factor=-10, max_lr_factor=0):

    samples = sample_exp_float(n_samples, base=10, min_factor=min_lr_factor, max_factor=max_lr_factor)

    return samples
