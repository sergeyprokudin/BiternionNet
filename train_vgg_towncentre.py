import numpy as np
import keras
import tensorflow as tf
import os
import sys
import yaml
import shutil

from models import vgg
from utils.angles import deg2bit, bit2deg
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.towncentre import load_towncentre
from utils.experiements import get_experiment_id, set_logging
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np, von_mises_neg_log_likelihood_keras
from utils.custom_keras_callbacks import SideModelCheckpoint

from scipy.stats import sem

from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate


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


def finetune_kappa(x, y_bit, model):
    ytr_preds_bit = model.predict(x)
    kappa_vals = np.arange(0, 10, 0.1)
    log_likelihoods = np.zeros(kappa_vals.shape)
    for i, kappa_val in enumerate(kappa_vals):
        kappa_preds = np.ones([x.shape[0], 1]) * kappa_val
        log_likelihoods[i] = von_mises_log_likelihood_np(y_bit, ytr_preds_bit, kappa_preds, input_type='biternion')
        print("kappa: %f, log-likelihood: %f" %(kappa_val, log_likelihoods[i]))
    max_ix = np.argmax(log_likelihoods)
    kappa = kappa_vals[max_ix]
    return kappa


def train():

    config_path = sys.argv[1]

    with open(config_path, 'r') as f:
        config = yaml.load(f)
    root_log_dir = config['root_log_dir']
    data_path = config['data_path']

    experiment_name = '_'.join([config['experiment_name'], get_experiment_id()])

    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)
    experiment_dir = os.path.join(root_log_dir, experiment_name)
    os.mkdir(experiment_dir)
    shutil.copy(config_path, experiment_dir)

    xtr, ytr_deg, xval, yval_deg, xte, yte_deg = load_towncentre(data_path, canonical_split=config['canonical_split'])

    image_height, image_width = xtr.shape[1], xtr.shape[2]
    ytr_bit = deg2bit(ytr_deg)
    yval_bit = deg2bit(yval_deg)
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
        yval = yval_bit
        yte = yte_bit
    elif net_output == 'degrees':
        y_pred = Dense(1)(vgg_x)
        metrics = ['mae']
        ytr = ytr_deg
        yval = yval_deg
        yte = yte_deg
    else:
        raise ValueError("net_output should be 'biternion' or 'degrees'")

    kappa = config['kappa']
    if kappa == 0.0:
        kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(Dense(256)(vgg_x)))
        # kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(vgg_x))
        model = Model(X, concatenate([y_pred, kappa_pred]))
        # kappa_model = Model(X, kappa_pred)
    else:
        model = Model(X, y_pred)

    custom_objects = {}

    if config['loss'] == 'cosine':
        print("using cosine loss..")
        loss_te = cosine_loss_tf
        custom_objects.update({'cosine_loss_tf': cosine_loss_tf})
    elif config['loss'] == 'von_mises':
        print("using von-mises loss..")
        loss_te = von_mises_loss_tf
        custom_objects.update({'von_mises_loss_tf': von_mises_loss_tf})
    elif config['loss'] == 'mad':
        print("using mad loss..")
        loss_te = mad_loss_tf
        custom_objects.update({'mad_loss_tf': mad_loss_tf})
    elif config['loss'] == 'vm_likelihood':
        print("using likelihood loss..")
        if kappa == 0.0:
            loss_te = von_mises_neg_log_likelihood_keras
            custom_objects.update({'von_mises_neg_log_likelihood_keras': von_mises_neg_log_likelihood_keras})
        else:

            def _von_mises_neg_log_likelihood_keras_fixed(y_true, y_pred):
                mu_pred = y_pred[:, 0:2]
                kappa_pred = tf.ones([tf.shape(y_pred[:, 2:])[0], 1])*kappa
                return -K.mean(von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred, input_type='biternion'))

            loss_te = _von_mises_neg_log_likelihood_keras_fixed
            custom_objects.update({'_von_mises_neg_log_likelihood_keras_fixed': _von_mises_neg_log_likelihood_keras_fixed})
    else:
        raise ValueError("loss should be 'mad','cosine','von_mises' or 'vm_likelihood'")

    optimizer = get_optimizer(config['optimizer_params'])

    model.compile(loss=loss_te,
                  optimizer=optimizer)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=experiment_dir,
                                                       write_images=True)

    train_csv_log = os.path.join(experiment_dir, 'train.csv')
    csv_callback = keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False)

    best_model_ckpt_file = os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.best_model.h5')

    model_ckpt_callback = keras.callbacks.ModelCheckpoint(best_model_ckpt_file,
                                                          monitor='val_loss',
                                                          mode='min',
                                                          save_best_only=True,
                                                          verbose=1)

    print("logs could be found at %s" % experiment_dir)

    model.fit(x=xtr, y=ytr,
              batch_size=config['batch_size'],
              epochs=config['n_epochs'],
              verbose=1,
              validation_data=(xval, yval),
              callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

    # model.save(os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.h5'))

    final_model_ckpt_file = os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.final_model.h5')
    model.save(final_model_ckpt_file)

    # model.load_weights(best_model_ckpt_file)
    model = load_model(best_model_ckpt_file, custom_objects=custom_objects)

    results = dict()
    results_yml_file = os.path.join(experiment_dir, 'results.yml')

    print("evaluating model..")

    def _eval_model(x, ytrue_deg, ytrue_bit, data_part):

        if net_output == 'biternion':
            ypreds = model.predict(x)
            ypreds_bit = ypreds[:, 0:2]
            ypreds_deg = bit2deg(ypreds_bit)
        elif net_output == 'degrees':
            ypreds = model.predict(x)
            ypreds_deg = ypreds[:, 0:1]
            ypreds_bit = deg2bit(ypreds_deg)

        if kappa == 0.0:
            kappa_preds = ypreds[:, 2:]
        else:
            kappa_preds =  np.ones([ytrue_deg.shape[0], 1]) * kappa

        loss = maad_from_deg(ypreds_deg, ytrue_deg)
        results['mean_loss_'+data_part] = float(np.mean(loss))
        results['std_loss_'+data_part] = float(np.std(loss))
        print("MAAD error (%s) : %f ± %f" % (data_part,
                                             results['mean_loss_'+data_part],
                                             results['std_loss_'+data_part]))

        results['mean_kappa_'+data_part] = float(np.mean(kappa_preds))
        results['std_kappa_'+data_part] = float(np.std(kappa_preds))
        print("predicted kappa (%s) : %f ± %f" % (data_part,
                                                      results['mean_kappa_'+data_part],
                                                      results['std_kappa_'+data_part]))

        log_likelihoods = von_mises_log_likelihood_np(ytrue_bit, ypreds_bit, kappa_preds,
                                                      input_type='biternion')
        results['log_likelihood_mean_'+data_part] = float(np.mean(log_likelihoods))
        results['log_likelihood_sem_'+data_part] = float(sem(log_likelihoods, axis=None))
        print("log-likelihood (%s) : %f ± %fSEM" % (data_part,
                                                    results['log_likelihood_mean_'+data_part],
                                                    results['log_likelihood_sem_'+data_part]))

    _eval_model(xtr, ytr_deg, ytr_bit, 'train')
    _eval_model(xval, yval_deg, yval_bit, 'validation')
    _eval_model(xte, yte_deg, yte_bit, 'test')

    print("stored model available at %s" % experiment_dir)

    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)

    return


if __name__ == '__main__':
    train()