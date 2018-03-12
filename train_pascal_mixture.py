import numpy as np
import os
import h5py
import binascii
import pandas as pd

from keras import backend as K
from models.biternion_mixture import BiternionMixture


PASCAL_DATA_DB = '/home/sprokudin/biternionnet/data/pascal_imagenet_train_test.h5'
LOGS_PATH = '/home/sprokudin/biternionnet/logs'
N_TRIALS = 20

PASCAL_CLASSES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                  'chair', 'diningtable', 'motorbike', 'sofa',  'train', 'tvmonitor']


def train_val_split(x, y, val_split=0.2, canonical_split=True):

    if canonical_split:
        val_split = 0.2
        np.random.seed(13)

    n_samples = x.shape[0]

    shuffled_samples = np.random.choice(n_samples, n_samples, replace=False)

    n_train = int((1-val_split)*n_samples)
    train_samples = shuffled_samples[0:n_train]
    val_samples = shuffled_samples[n_train:]

    x_train, y_train = x[train_samples], y[train_samples]
    x_val, y_val = x[val_samples], y[val_samples]

    np.random.seed(None)

    return x_train, y_train, x_val, y_val


def get_class_data(data_h5, cls_name):

    images = np.asarray(data_h5[cls_name]['images'])
    azimuth_bit = np.asarray(data_h5[cls_name]['azimuth_bit'])
    elevation_bit = np.asarray(data_h5[cls_name]['elevation_bit'])
    tilt_bit = np.asarray(data_h5[cls_name]['tilt_bit'])
    angles = np.hstack([azimuth_bit, elevation_bit, tilt_bit])

    return images, angles


def merge_all_classes(data):

    images = []
    angles = []
    for cls_key in data.keys():
        cls_images, cls_angles = get_class_data(data, cls_key)
        images.append(cls_images)
        angles.append(cls_angles)

    images = np.vstack(images)
    angles = np.vstack(angles)

    return images, angles


def load_pascal_data(cls=None, val_split=0.2):

    train_test_data_db = h5py.File(PASCAL_DATA_DB, 'r')

    train_data = train_test_data_db['train']
    test_data = train_test_data_db['test']

    if cls is None:
        x_train, y_train = merge_all_classes(train_data)
        x_test, y_test = merge_all_classes(test_data)
    else:
        x_train, y_train = get_class_data(train_data, cls)
        x_test, y_test = get_class_data(test_data, cls)

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_split=val_split)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_experiment_id():
    experiment_id = binascii.hexlify(os.urandom(10))
    return experiment_id.decode("utf-8")


def select_params():

    params ={}
    params['lr'] = np.random.choice([1.0e-3, 1.0e-4, 1.0e-5])
    params['batch_size'] = np.random.choice([8, 16, 32, 128])
    params['hlayer_size'] = np.random.choice([256, 512, 1024])

    return params


def fixed_params():

    params ={}
    params['lr'] = 1.0e-4
    params['batch_size'] = 32
    params['hlayer_size'] = 512
    params['z_size'] = 8
    params['n_samples'] = 50

    return params


def train_model(class_name):

    global_results_log = '/home/sprokudin/biternionnet/logs/V2_bimixture_%s.csv' % (class_name)

    if not os.path.exists(global_results_log):
        with open(global_results_log, 'w') as f:
            f.write("checkpoint_path;train_maad;train_ll;val_maad;val_ll;test_maad;test_ll;az_kappa;el_kappa;ti_kappa\n")

    x_train, y_train, x_val, y_val, x_test, y_test = load_pascal_data(cls=class_name, val_split=0.2)

    print("TRAINING on CLASS : %s" % class_name)

    exp_id = get_experiment_id()
    params = fixed_params()
    # params = select_params()

    K.clear_session()

    ckpt_name = 'bimixture_%s_%s_hls%d_zs_%d_ns_%d.1e.h5' % (class_name, exp_id, params['hlayer_size'],
                                                             params['z_size'], params['n_samples'])
    ckpt_path = os.path.join(LOGS_PATH, ckpt_name)

    K.clear_session()
    #20 samples, z_size=8
    model = BiternionMixture(input_shape=x_train.shape[1:], debug=False,
                             n_samples=params['n_samples'], z_size=params['z_size'],
                             learning_rate=params['lr'], hlayer_size=params['hlayer_size'])

    train_maad, train_ll, val_maad, val_ll, test_maad, test_ll = \
        model.train_finetune_eval(x_train, y_train, x_val, y_val, x_test, y_test,
                                  ckpt_path, batch_size=params['batch_size'], patience=20, epochs=100)

    with open(global_results_log, 'a') as f:
        res_str = '%s;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f'\
                  % (ckpt_path, train_maad, train_ll, val_maad, val_ll, test_maad, test_ll,
                     0, 0, 0)
        f.write("%s\n" % res_str)

    print("Trial finished. Best model saved at %s" % ckpt_path)

    return


def main():

    for i in range(0, N_TRIALS):
        class_name = "aeroplane" #np.random.choice(PASCAL_CLASSES)
        train_model(class_name)

    print("Fin.")

    return

if __name__ == '__main__':
    main()