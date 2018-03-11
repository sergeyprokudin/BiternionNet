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


best_cosine_models = \
    {'aeroplane': '/home/sprokudin/biternionnet/logs/bicnn_cosine_aeroplane_50f9b5922829ca8310da_bs32_hls512_lr_1.0e-04.h5',
     'bicycle': '/home/sprokudin/biternionnet/logs/bicnn_cosine_bicycle_fb2a1e5a98229a2b9624_bs32_hls512_lr_1.0e-04.h5',
 'boat': '/home/sprokudin/biternionnet/logs/bicnn_cosine_boat_56fe023b5a6776ce10c5_bs32_hls512_lr_1.0e-04.h5',
 'bottle': '/home/sprokudin/biternionnet/logs/bicnn_cosine_bottle_e40e75bf87563bdb64e8_bs32_hls512_lr_1.0e-04.h5',
 'bus': '/home/sprokudin/biternionnet/logs/bicnn_cosine_bus_08863531002b27f2b2b9_bs32_hls512_lr_1.0e-04.h5',
 'car': '/home/sprokudin/biternionnet/logs/bicnn_cosine_car_b9aa672e18bd0997d441_bs32_hls512_lr_1.0e-04.h5',
 'chair': '/home/sprokudin/biternionnet/logs/bicnn_cosine_chair_cdd72ff9ee65afb6cd13_bs32_hls512_lr_1.0e-04.h5',
 'diningtable': '/home/sprokudin/biternionnet/logs/bicnn_cosine_diningtable_81032fcde707bff07749_bs32_hls512_lr_1.0e-04.h5',
 'motorbike': '/home/sprokudin/biternionnet/logs/bicnn_cosine_motorbike_cc12ced68b301670f873_bs32_hls512_lr_1.0e-04.h5',
 'sofa': '/home/sprokudin/biternionnet/logs/bicnn_cosine_sofa_ae299773b9140ed37b4e_bs32_hls512_lr_1.0e-04.h5',
 'train': '/home/sprokudin/biternionnet/logs/bicnn_cosine_train_1c245de856d7bd05889a_bs32_hls512_lr_1.0e-04.h5',
 'tvmonitor': '/home/sprokudin/biternionnet/logs/bicnn_cosine_tvmonitor_c6a143e3666bb6041dd0_bs32_hls512_lr_1.0e-04.h5'}


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

    return params


def train_model(class_name, loss_type, pretrain=True):

    global_results_log = '/home/sprokudin/biternionnet/logs/V1_biternion_%s_%s.csv' % (loss_type, class_name)

    if not os.path.exists(global_results_log):
        with open(global_results_log, 'w') as f:
            f.write("checkpoint_path;train_maad;train_ll;val_maad;val_ll;test_maad;test_ll;az_kappa;el_kappa;ti_kappa\n")

    x_train, y_train, x_val, y_val, x_test, y_test = load_pascal_data(cls=class_name, val_split=0.2)

    print("TRAINING on CLASS : %s" % class_name)
    print("LOSS function used: %s" % loss_type)

    exp_id = get_experiment_id()
    params = fixed_params()
    # params = select_params()

    K.clear_session()

    ckpt_name = 'bicnn_%s_%s_%s_bs%d_hls%d_lr_%0.1e.h5' % (loss_type, class_name, exp_id, params['batch_size'], params['hlayer_size'], params['lr'])
    ckpt_path = os.path.join(LOGS_PATH, ckpt_name)

    if loss_type == 'likelihood' and pretrain:
        print("Pre-training model with fixed kappas..")
        model = BiternionCNN(input_shape=x_train.shape[1:], debug=False, loss_type='cosine',
                             learning_rate=params['lr'], hlayer_size=params['hlayer_size'])

        train_maad, train_ll, val_maad, val_ll, test_maad, test_ll, kappas = \
            model.train_finetune_eval(x_train, y_train, x_val, y_val, x_test, y_test,
                                  ckpt_path, batch_size=params['batch_size'], patience=5, epochs=20)

    K.clear_session()
    model = BiternionCNN(input_shape=x_train.shape[1:], debug=False, loss_type=loss_type,
                         learning_rate=params['lr'], hlayer_size=params['hlayer_size'])

    if loss_type == 'likelihood' and pretrain:
          model.model.load_weights(ckpt_path)
          # model.model.load_weights(best_cosine_models[class_name])

    train_maad, train_ll, val_maad, val_ll, test_maad, test_ll, kappas = \
        model.train_finetune_eval(x_train, y_train, x_val, y_val, x_test, y_test,
                                  ckpt_path, batch_size=params['batch_size'], patience=5, epochs=100)

    with open(global_results_log, 'a') as f:
        res_str = '%s;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f;%2.2f'\
                  % (ckpt_path, train_maad, train_ll, val_maad, val_ll, test_maad, test_ll,
                     kappas[0], kappas[1], kappas[2])
        f.write("%s\n" % res_str)

    print("Trial finished. Best model saved at %s" % ckpt_path)

    return


def main():

    for i in range(0, N_TRIALS):

        class_name = np.random.choice(PASCAL_CLASSES)
        loss_type = np.random.choice(['cosine', 'likelihood'])
        pretrain = np.random.choice([False, True])
        train_model(class_name, loss_type, pretrain=pretrain)

    print("Fin.")

    return

if __name__ == '__main__':
    main()