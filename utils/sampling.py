import numpy as np


def sample_multiple_gauassians_np(mu, std, n_samples=10):
    """Sample points from multiple multivariate gaussian distributions

    Parameters
    ----------

    mu: numpy array of shape [n_points, n_dims]
        mean values of multiple multivariate gaussians
    std: numpy array of shape [n_points, n_dims]
        stdev values of multiple multivariate gaussians
    n_samples: int
        number of samples to draw from each distribution

    Returns
    -------

    samples: numpy array of shape [n_points, n_samples, n_dims]
        samples from each gaussian
    """
    n_points, n_dims = mu.shape

    eps = np.random.normal(size=[n_points, n_samples, n_dims])

    mu_tiled = np.tile(mu.reshape([n_points, 1, n_dims]), [1, n_samples, 1])
    std_tiled = np.tile(std.reshape([n_points, 1, n_dims]), [1, n_samples, 1])

    samples = mu_tiled + eps*std_tiled

    return samples
