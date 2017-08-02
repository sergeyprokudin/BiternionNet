import numpy as np


def deg2bit(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


def bit2deg(angles_bit):
    return (np.rad2deg(np.arctan2(angles_bit[:, 1], angles_bit[:, 0])) + 360) % 360


def bit2deg_multi(angles_bit):
    """ Convert biternion representation to degree for multiple samples

    Parameters
    ----------
    angles_bit: numpy array of shape [n_points, n_predictions, 2]
        multiple predictions

    Returns
    -------

    deg_angles: numpy array of shape [n_points, n_predictions]
        multiple predictions converted to degree representation
    """

    deg_angles = np.asarray([bit2deg(angles_bit[:, i, :]) for i in range(0, angles_bit.shape[1])]).T

    return deg_angles
