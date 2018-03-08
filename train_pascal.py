import numpy as np
import os
import h5py
import binascii
import pandas as pd

from keras import backend as K
from models.biternion_cnn import BiternionCNN


PASCAL_DATA_DB = '/home/sprokudin/biternionnet/data/pascal_imagenet_train_test.h5'
LOGS_PATH = '/home/sprokudin/biternionnet/logs'
GLOBAL_RESULTS_LOG = '/home/sprokudin/biternionnet/logs/biternion_cosine_loss.csv'
N_TRIALS = 5
LOSS_TYPE = 'cosine'


def train_val_split(x, y, val_split=0.2):

    n_samples = x.shape[0]

    shuffled_samples = np.random.choice(n_samples, n_samples, replace=False)
    n_train = int((1-val_split)*n_samples)
    train_samples = shuffled_samples[0:n_train]
    val_samples = shuffled_samples[n_train:]

    x_train, y_train = x[train_samples], y[train_samples]
    x_val, y_val = x[val_samples], y[val_samples]

    return x_train, y_train, x_val, y_val


def merge_all_classes(data):

    images = []
    angles = []
    for obj_key in data.keys():
        images.append(np.asarray(data[obj_key]['images']))
        azimuth_bit = np.asarray(data[obj_key]['azimuth_bit'])
        elevation_bit = np.asarray(data[obj_key]['elevation_bit'])
        tilt_bit = np.asarray(data[obj_key]['tilt_bit'])
        angles.append(np.hstack([azimuth_bit, elevation_bit, tilt_bit]))

    images = np.vstack(images)
    angles = np.vstack(angles)

    return images, angles


def load_data(val_split=0.2):

    train_test_data_db = h5py.File(PASCAL_DATA_DB, 'r')
    train_data = train_test_data_db['train']
    # test_data = train_test_data_db['test']

    x_train, y_train = merge_all_classes(train_data)

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_split=val_split)

    return x_train, y_train, x_val, y_val


def get_experiment_id():
    experiment_id = binascii.hexlify(os.urandom(10))
    return experiment_id.decode("utf-8")


def select_params():

    params ={}
    params['lr'] = np.random.choice([1.0e-3, 1.0e-4, 1.0e-5])
    params['batch_size'] = np.random.choice([8, 16, 32])
    params['hlayer_size'] = np.random.choice([256, 512, 1024])

    return params


def main():

    x_train, y_train, x_val, y_val = load_data(val_split=0.2)

    if not os.path.exists(GLOBAL_RESULTS_LOG):
        with open(GLOBAL_RESULTS_LOG, 'w') as f:
            f.write("checkpoint_path;validation_loss\n")

    for i in range(0, N_TRIALS):
        exp_id = get_experiment_id()
        params = select_params()

        K.clear_session()
        model = BiternionCNN(input_shape=x_train.shape[1:], debug=False, loss_type='cosine',
                             learning_rate=params['lr'], hlayer_size=params['hlayer_size'])

        ckpt_name = 'bicnn_%s_bs%d_hls%d_lr_%0.1e' % (exp_id, params['batch_size'], params['hlayer_size'], params['lr'])
        ckp_path = os.path.join(LOGS_PATH, ckpt_name)
        model.fit(x_train, y_train, [x_val, y_val], epochs=50, ckpt_path=ckp_path,
                  patience=5, batch_size=params['batch_size'])
        val_loss = model.model.evaluate(x_val, y_val)
        with open(GLOBAL_RESULTS_LOG, 'a') as f:
            f.write("%s;%f\n" % (ckp_path, val_loss))

    print("model")

    return


if __name__ == '__main__':
    main()