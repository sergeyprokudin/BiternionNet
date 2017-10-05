import numpy as np
from utils.angles import rad2bit, deg2bit
from utils.idiap import load_idiap_part
from utils.caviar import load_caviar


def load_dataset(config):

    if config['dataset'] == 'IDIAP':

        (xtr, ytr_rad), (xval, yval_rad), (xte, yte_rad) = load_idiap_part(config['data_path'],
                                                                           config['net_output'])

        ytr_deg = np.rad2deg(ytr_rad)
        yval_deg = np.rad2deg(yval_rad)
        yte_deg = np.rad2deg(yte_rad)

    elif (config['dataset'] == 'CAVIAR-o') or (config['dataset'] == 'CAVIAR-c'):

        (xtr, ytr_deg), (xval, yval_deg), (xte, yte_deg) = load_caviar(config['data_path'])

    else:

        raise ValueError("invalid dataset name!")

    ytr_bit = deg2bit(ytr_deg)
    yval_bit = deg2bit(yval_deg)
    yte_bit = deg2bit(yte_deg)

    return (xtr, ytr_bit, ytr_deg), (xval, yval_bit, yval_deg), (xte, yte_bit, yte_deg)


def load_dataset1(name, data_path, part=None):

    if name == 'IDIAP':

        (xtr, ytr_rad), (xval, yval_rad), (xte, yte_rad) = load_idiap_part(data_path,
                                                                           part)

        ytr_deg = np.rad2deg(ytr_rad)
        yval_deg = np.rad2deg(yval_rad)
        yte_deg = np.rad2deg(yte_rad)

    elif (name == 'CAVIAR-o') or (name == 'CAVIAR-c'):

        (xtr, ytr_deg), (xval, yval_deg), (xte, yte_deg) = load_caviar(data_path)

    else:

        raise ValueError("invalid dataset name!")

    ytr_bit = deg2bit(ytr_deg)
    yval_bit = deg2bit(yval_deg)
    yte_bit = deg2bit(yte_deg)

    return (xtr, ytr_bit, ytr_deg), (xval, yval_bit, yval_deg), (xte, yte_bit, yte_deg)
