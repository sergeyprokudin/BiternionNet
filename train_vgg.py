import numpy as np
import keras
import tensorflow as tf
import os
import sys
import yaml
import shutil
import itertools
import pandas as pd

from models import vgg
from utils.angles import rad2bit
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf
from utils.idiap import load_idiap_part
from utils.experiements import get_experiment_id
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np, von_mises_neg_log_likelihood_keras

from keras import backend as K


def load_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config

def load_dataset(config):

    if config['dataset'] == 'IDIAP':

        (xtr, ytr_rad), (xval, yval_rad), (xte, yte_rad) = load_idiap_part(config['data_path'],
                                                                           config['net_output'])
    else:

        raise ValueError("invalid dataset name!")

    ytr_bit = rad2bit(ytr_rad)
    yval_bit = rad2bit(yval_rad)
    yte_bit = rad2bit(yte_rad)
    ytr_deg = np.rad2deg(ytr_rad)
    yval_deg = np.rad2deg(yval_rad)
    yte_deg = np.rad2deg(yte_rad)

    return (xtr, ytr_bit, ytr_deg), (xval, yval_bit, yval_deg), (xte, yte_bit, yte_deg)


def pick_loss(config):

    if config['loss'] == 'cosine':
        print("using cosine loss..")
        loss_te = cosine_loss_tf
    elif config['loss'] == 'von_mises':
        print("using von-mises loss..")
        loss_te = von_mises_loss_tf
    elif config['loss'] == 'mad':
        print("using mad loss..")
        loss_te = mad_loss_tf
    elif config['loss'] == 'vm_likelihood':
        print("using likelihood loss..")
        if config['predict_kappa']:
            loss_te = von_mises_neg_log_likelihood_keras
        else:

            def _von_mises_neg_log_likelihood_keras_fixed(y_true, y_pred):
                mu_pred = y_pred[:, 0:2]
                kappa_pred = tf.ones([tf.shape(y_pred[:, 2:])[0], 1])*config['fixed_kappa_value']
                return -K.mean(von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred))

            loss_te = _von_mises_neg_log_likelihood_keras_fixed
    else:
        raise ValueError("loss should be 'mad','cosine','von_mises' or 'vm_likelihood'")

    return loss_te


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


def make_lr_batch_size_grid():

    max_lr = 1.0
    lr_step = 0.1
    min_lr_factor = 10
    possible_learning_rates = np.asarray([max_lr * lr_step ** (n - 1) for n in range(1, min_lr_factor + 1)])

    min_batch_size = 4
    bs_step = 2
    max_size_factor = 8
    possible_batch_sizes = np.asarray([min_batch_size * bs_step ** (n - 1) for n in range(1, max_size_factor + 1)])

    grid = list(itertools.product(possible_learning_rates, possible_batch_sizes))

    return grid


def finetune_kappa(x, y_bit, model):
    ytr_preds_bit = model.predict(x)
    kappa_vals = np.arange(0, 1000, 1.0)
    log_likelihoods = np.zeros(kappa_vals.shape)
    for i, kappa_val in enumerate(kappa_vals):
        kappa_preds = np.ones([x.shape[0], 1]) * kappa_val
        log_likelihoods[i] = von_mises_log_likelihood_np(y_bit, ytr_preds_bit, kappa_preds, input_type='biternion')
        print("kappa: %f, log-likelihood: %f" %(kappa_val, log_likelihoods[i]))
    max_ix = np.argmax(log_likelihoods)
    kappa = kappa_vals[max_ix]
    return kappa


def train():

    if len(sys.argv) != 2:
        print("Ivalid number of params! Usage: python train_vgg.py config_path")

    config = load_config(sys.argv[1])

    root_log_dir = config['root_log_dir']

    if not os.path.exists(root_log_dir):
        os.makedirs(root_log_dir, exist_ok=True)

    experiment_name = '_'.join([config['experiment_name'], get_experiment_id()])

    experiment_dir = os.path.join(root_log_dir, experiment_name)

    os.mkdir(experiment_dir)

    (xtr, ytr_bit, ytr_deg), (xval, yval_bit, yval_deg), (xte, yte_bit, yte_deg) = load_dataset(config)

    loss_te = pick_loss(config)

    image_height, image_width, n_channels = xtr.shape[1], xtr.shape[2], xtr.shape[3]

    predict_kappa = config['predict_kappa']
    fixed_kappa_value = config['fixed_kappa_value']

    best_trial_id = 0

    n_trials = config['n_trials']
    batch_sizes = config['batch_sizes']
    learning_rates = config['learning_rates']
    params_grid = list(itertools.product(learning_rates, batch_sizes))*n_trials

    results = dict()
    res_cols = ['trial_id', 'batch_size', 'learning_rate', 'val_maad', 'val_likelihood', 'test_maad', 'test_likelihood']
    results_df = pd.DataFrame(columns=res_cols)
    results_csv = os.path.join(experiment_dir, 'results.csv')

    for tid, params in enumerate(params_grid):

        learning_rate = params[0]
        batch_size = params[1]
        print("TRIAL %d // %d" % (tid, n_trials))
        print("batch_size: %d" % batch_size)
        print("learning_rate: %f" % learning_rate)

        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)
        print("logs could be found at %s" % trial_dir)

        vgg_model = vgg.BiternionVGG(image_height=image_height,
                                     image_width=image_width,
                                     n_channels=n_channels,
                                     predict_kappa=predict_kappa,
                                     fixed_kappa_value=fixed_kappa_value)

        optimizer = keras.optimizers.Adam(epsilon=1.0e-07,
                                          lr=learning_rate,
                                          decay=0.0)

        vgg_model.model.compile(loss=loss_te, optimizer=optimizer)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=trial_dir)

        train_csv_log = os.path.join(trial_dir, 'train.csv')
        csv_callback = keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False)

        best_model_weights_file = os.path.join(trial_dir, 'vgg_bit_' + config['loss'] + '_town.best.weights.h5')

        model_ckpt_callback = keras.callbacks.ModelCheckpoint(best_model_weights_file,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              period=1,
                                                              verbose=1)

        vgg_model.model.save_weights(best_model_weights_file)

        vgg_model.model.fit(x=xtr, y=ytr_bit,
                            batch_size=batch_size,
                            epochs=config['n_epochs'],
                            verbose=1,
                            validation_data=(xval, yval_bit),
                            callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

        best_model = vgg.BiternionVGG(image_height=image_height,
                                      image_width=image_width,
                                      n_channels=n_channels,
                                      predict_kappa=predict_kappa,
                                      fixed_kappa_value=fixed_kappa_value)

        best_model.model.load_weights(best_model_weights_file)

        trial_results = dict()

        trial_results['learning_rate'] = float(learning_rate)
        trial_results['batch_size'] = float(batch_size)
        trial_results['ckpt_path'] = best_model_weights_file
        trial_results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
        trial_results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
        trial_results['test'] = best_model.evaluate(xte, yte_deg, 'test')
        results[tid] = trial_results

        results_np = np.asarray([tid, batch_size, learning_rate,
                                 trial_results['validation']['maad_loss'],
                                 trial_results['validation']['log_likelihood_mean'],
                                 trial_results['test']['maad_loss'],
                                 trial_results['test']['log_likelihood_mean']]).reshape([1, 7])

        trial_res_df = pd.DataFrame(results_np, columns=res_cols)
        results_df = results_df.append(trial_res_df)
        results_df.to_csv(results_csv)

        if tid > 0:
            if trial_results['validation']['log_likelihood_mean'] > \
                    results[best_trial_id]['validation']['log_likelihood_mean']:
                best_trial_id = tid
                print("Better validation loss achieved, current best trial: %d" % best_trial_id)

    print("loading best model..")
    best_ckpt_path = results[best_trial_id]['ckpt_path']
    overall_best_ckpt_path = os.path.join(experiment_dir, 'vgg.full_model.overall_best.weights.hdf5')
    shutil.copy(best_ckpt_path, overall_best_ckpt_path)

    best_model = vgg.BiternionVGG(image_height=image_height,
                                  image_width=image_width,
                                  n_channels=n_channels,
                                  predict_kappa=predict_kappa,
                                  fixed_kappa_value=fixed_kappa_value)

    best_model.model.load_weights(overall_best_ckpt_path)

    print("finetuning kappa values..")
    best_kappa = fixed_kappa_value
    if not predict_kappa:
        best_kappa = finetune_kappa(xval, yval_bit, best_model)
        print("best kappa: %f" % best_kappa)

    best_model = vgg.BiternionVGG(image_height=image_height,
                                  image_width=image_width,
                                  n_channels=n_channels,
                                  predict_kappa=predict_kappa,
                                  fixed_kappa_value=best_kappa)

    best_model.model.load_weights(overall_best_ckpt_path)

    best_results = dict()
    best_results['learning_rate'] = results[best_trial_id]['learning_rate']
    best_results['batch_size'] = results[best_trial_id]['batch_size']

    print("evaluating best model..")
    best_results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
    best_results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
    best_results['test'] = best_model.evaluate(xte, yte_deg, 'test')
    results['best'] = best_results

    results_yml_file = os.path.join(experiment_dir, 'results.yml')
    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)

    results_df.to_csv(results_csv)

    return


if __name__ == '__main__':
    train()