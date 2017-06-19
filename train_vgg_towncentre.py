import numpy as np
import keras
import tensorflow as tf
import os
import yaml
import shutil

from models import vgg
from utils.angles import deg2bit, bit2deg
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.towncentre import load_towncentre
from utils.experiements import get_experiment_id, set_logging
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np

from keras.layers import Input, Dense, Lambda
import keras.backend as K
from keras.models import Model


def get_optimizer(optimizer_params):
    if optimizer_params['name'] == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(rho=optimizer_params['rho'],
                                              epsilon=optimizer_params['epsilon'],
                                              lr=optimizer_params['learning_rate'],
                                              decay=optimizer_params['decay'])
    elif optimizer_params['name'] == 'Adam':
        optimizer = keras.optimizers.Adam(epsilon=optimizer_params['epsilon'],
                                          lr=optimizer_params['learning_rate'],
                                          decay=optimizer_params['decay'])
    return optimizer


def train():

    config_path = 'train_vgg_towncentre.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    root_log_dir = config['root_log_dir']
    data_path = config['data_path']

    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)
    experiment_id = get_experiment_id()
    experiment_dir = os.path.join(root_log_dir, str(experiment_id))
    os.mkdir(experiment_dir)
    shutil.copy(config_path, experiment_dir)

    xtr, ytr_deg, xte, yte_deg = load_towncentre(data_path, canonical_split=config['canonical_split'])
    image_height, image_width = xtr.shape[1], xtr.shape[2]
    ytr_bit = deg2bit(ytr_deg)
    yte_bit = deg2bit(yte_deg)

    X = Input(shape=[50, 50, 3])

    vgg_x = vgg.vgg_model(final_layer=False,
                          image_height=image_height,
                          image_width=image_width)(X)

    net_output = config['net_output']

    if net_output == 'biternion':
        y_pred = Lambda(lambda x: K.l2_normalize(x, axis=1))(Dense(2)(vgg_x))
        metrics = ['cosine']
        ytr = ytr_bit
        yte = yte_bit
    elif net_output == 'degrees':
        y_pred = Dense(1)(vgg_x)
        metrics = ['mae']
        ytr = ytr_deg
        yte = yte_deg
    else:
        raise ValueError("net_output should be 'biternion' or 'degrees'")

    model = Model(X, y_pred)

    kappa = config['kappa']
    if kappa == 0.0:
        kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(vgg_x))
        kappa_model = Model(X, kappa_pred)
    else:
        kappa_pred = tf.ones([tf.shape(y_pred)[0], 1])*kappa

    if config['loss'] == 'cosine':
        loss = cosine_loss_tf
    elif config['loss'] == 'von_mises':
        loss = von_mises_loss_tf
    elif config['loss'] == 'mad':
        loss = mad_loss_tf
    elif config['loss'] == 'vm_likelihood':

        def _von_mises_neg_log_likelihood_keras(y_true, y_pred):
            return -von_mises_log_likelihood_tf(y_true, y_pred, kappa_pred, input_type='biternion')

        loss = _von_mises_neg_log_likelihood_keras
    else:
        raise ValueError("loss should be 'mad','cosine','von_mises' or 'vm_likelihood'")

    optimizer = get_optimizer(config['optimizer_params'])

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    validation_split = config['validation_split']
    if validation_split == 0:
        print("Using test data as validation..")
        model.fit(x=xtr, y=ytr,
                  batch_size=config['batch_size'],
                  epochs=config['n_epochs'],
                  verbose=1,
                  validation_data=(xte, yte))
    else:
        model.fit(x=xtr, y=ytr,
                  batch_size=config['batch_size'],
                  epochs=config['n_epochs'],
                  verbose=1,
                  validation_split=validation_split)
        # print("fine-tuning the model..")
        # model.optimizer.lr.assign(config['optimizer_params']['learning_rate']*0.01)

    model.save(os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.h5'))

    if net_output == 'biternion':
        yte_preds_bit = model.predict(xte)
        yte_preds_deg = bit2deg(yte_preds_bit)

    elif net_output == 'degrees':
        yte_preds_deg = np.squeeze(model.predict(xte))

    loss = maad_from_deg(yte_preds_deg, yte_deg)
    mean_loss = np.mean(loss)
    std_loss = np.std(loss)

    print("MAAD error (test) : %f ± %f" % (mean_loss, std_loss))

    if kappa == 0.0:
        kappa_preds = kappa_model.predict(xte)
    else:
        kappa_preds = np.ones([xte.shape[0], 1]) * kappa

    if net_output == 'biternion':
        log_likelihood_loss = von_mises_log_likelihood_np(yte_bit, yte_preds_bit, kappa_preds)
        print("log-likelihood (test) : %f" % log_likelihood_loss)

    print("stored model available at %s" % experiment_dir)

    return


if __name__ == '__main__':
    train()