import numpy as np
import os
import h5py
import binascii
import pandas as pd

from keras import backend as K
from models.biternion_cnn import BiternionCNN


PASCAL_DATA_DB = '/home/sprokudin/biternionnet/data/pascal_imagenet_train_test.h5'
LOGS_PATH = '/home/sprokudin/biternionnet/logs'
N_TRIALS = 20

PASCAL_CLASSES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                  'chair', 'diningtable', 'motorbike', 'sofa',  'train', 'tvmonitor']


def train_val_split(x, y, val_split=0.2):

    n_samples = x.shape[0]

    shuffled_samples = np.random.choice(n_samples, n_samples, replace=False)
    n_train = int((1-val_split)*n_samples)
    train_samples = shuffled_samples[0:n_train]
    val_samples = shuffled_samples[n_train:]

    x_train, y_train = x[train_samples], y[train_samples]
    x_val, y_val = x[val_samples], y[val_samples]

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


def load_data(cls=None, val_split=0.2):

    train_test_data_db = h5py.File(PASCAL_DATA_DB, 'r')

    train_data = train_test_data_db['train']

    if cls is None:
        x_train, y_train = merge_all_classes(train_data)
    else:
        x_train, y_train = get_class_data(train_data, cls)

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train, val_split=val_split)

    return x_train, y_train, x_val, y_val


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

    return params


def train_model(class_name, loss_type):

    global_results_log = '/home/sprokudin/biternionnet/logs/biternion_%s_%s.csv' % (loss_type, class_name)

    if not os.path.exists(global_results_log):
        with open(global_results_log, 'w') as f:
            f.write("checkpoint_path;validation_loss\n")

    x_train, y_train, x_val, y_val = load_data(cls=class_name, val_split=0.2)

    print("TRAINING on CLASS : %s" % class_name)
    print("LOSS function used: %s" % loss_type)

    exp_id = get_experiment_id()
    params = fixed_params()
    # params = select_params()

    K.clear_session()
    model = BiternionCNN(input_shape=x_train.shape[1:], debug=False, loss_type=loss_type,
                         learning_rate=params['lr'], hlayer_size=params['hlayer_size'])

    ckpt_name = 'bicnn_%s_%s_%s_bs%d_hls%d_lr_%0.1e.h5' % (loss_type, class_name, exp_id, params['batch_size'], params['hlayer_size'], params['lr'])
    ckp_path = os.path.join(LOGS_PATH, ckpt_name)
    model.fit(x_train, y_train, [x_val, y_val], epochs=200, ckpt_path=ckp_path,
              patience=10, batch_size=params['batch_size'])
    val_loss = model.model.evaluate(x_val, y_val)
    with open(global_results_log, 'a') as f:
        f.write("%s;%f\n" % (ckp_path, val_loss))

    print("Trial finished. Best model saved at %s" % ckp_path)


    return


def main():

    for i in range(0, N_TRIALS):

        class_name = 'boat' #np.random.choice(PASCAL_CLASSES)
        loss_type = 'likelihood' #np.random.choice(['cosine', 'likelihood'])
        train_model(class_name, loss_type)

    print("Fin.")

    return

if __name__ == '__main__':
    main()