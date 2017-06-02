import numpy as np


def deg2bit(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


def bit2deg(angles_bit):
    return (np.rad2deg(np.arctan2(angles_bit[:,1], angles_bit[:,0])) + 360) % 360