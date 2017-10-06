import os
import sys
import shutil
import yaml
import itertools
import numpy as np
import pandas as pd
import keras

from models.cvae import CVAE
from utils.load_datasets import load_dataset
from utils.experiements import get_experiment_id
from utils.custom_keras_callbacks import ModelCheckpointEveryNBatch
from utils import hyper_tune as ht


def load_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config


def generate_hyper_params(n_trials):

    hyp_params = dict()
    hyp_params['learning_rate'] = ht.sample_exp_float(n_trials, base=10, min_factor=-5, max_factor=-3)
    hyp_params['batch_size'] = ht.sample_exp_int(n_trials, base=2, min_factor=1, max_factor=8)
    hyp_params['beta1'] = np.random.rand(n_trials)
    hyp_params['beta2'] = np.random.rand(n_trials)
    hyp_params['epsilons'] = ht.sample_exp_float(n_trials, base=10, min_factor=-6, max_factor=-8)
    hyp_params['conv_dropout'] = np.random.rand(n_trials)
    hyp_params['fc_dropout'] = np.random.rand(n_trials)
    hyp_params['n_hidden_units'] = ht.sample_exp_int(n_trials, base=2, min_factor=1, max_factor=4)
    hyp_params['vgg_fc_layer_size'] = ht.sample_exp_int(n_trials, base=2, min_factor=5, max_factor=10)
    hyp_params['cvae_fc_layer_size'] = ht.sample_exp_int(n_trials, base=2, min_factor=5, max_factor=10)

    return hyp_params


def load_hyper_params(config):

    hyp_params = dict()
    n_trials = config['n_trials']
    hyp_params['learning_rate'] = np.ones(n_trials)*config['learning_rate']
    hyp_params['batch_size'] = np.ones(n_trials, dtype=int)*config['batch_size']
    hyp_params['beta1'] = np.ones(n_trials)*config['beta1']
    hyp_params['beta2'] = np.ones(n_trials)*config['beta2']
    hyp_params['epsilons'] = np.ones(n_trials)*config['epsilon']
    hyp_params['conv_dropout'] = np.ones(n_trials)*config['conv_dropout']
    hyp_params['fc_dropout'] = np.ones(n_trials)*config['fc_dropout']
    hyp_params['vgg_fc_layer_size'] = np.ones(n_trials, dtype=int)*config['vgg_fc_layer_size']
    hyp_params['cvae_fc_layer_size'] = np.ones(n_trials, dtype=int)*config['cvae_fc_layer_size']
    hyp_params['n_hidden_units'] = np.ones(n_trials, dtype=int)*config['n_hidden_units']

    return hyp_params


def get_trial_hyp_params(hyp_params, trial_id):

    trial_hyp_dic = dict()

    for param in hyp_params.keys():
        trial_hyp_dic[param] = hyp_params[param][trial_id]

    return trial_hyp_dic


def print_hyp_params(hyp_params, tid):

    n_trials = len(hyp_params['learning_rate'])
    print("TRIAL %d // %d" % (tid, n_trials))

    for key, value in hyp_params.items():
        print("%s : %f" % (key, value[tid]))

    return


def evaluate_model(model, data):

    xtr, ytr_deg, xval, yval_deg, xte, yte_deg = data

    results = dict()
    results['train'] = model.evaluate(xtr, ytr_deg, 'train')
    results['validation'] = model.evaluate(xval, yval_deg, 'validation')
    results['test'] = model.evaluate(xte, yte_deg, 'test')

    return results


def results_to_df(trial_results, trial_hyp_params):

    dfs = []

    hyp_df = pd.DataFrame(trial_hyp_params, index=[0])

    dfs.append(hyp_df)

    for key in trial_results.keys():
        dfs.append(pd.DataFrame.from_dict(trial_results[key]).add_prefix(key+'_'))

    res_df = pd.concat(dfs, axis=1)

    return res_df


def define_callbacks(config, trial_dir, ckpt_path, val_data):

    callbacks = []

    xval, yval_bit = val_data

    if config['model_type'] == 'cvae':
        x = [xval, yval_bit]
    else:
        x = xval

    callbacks.append(keras.callbacks.TensorBoard(log_dir=trial_dir))

    train_csv_log = os.path.join(trial_dir, 'train.csv')
    callbacks.append(keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False))

    val_loss_log_path = os.path.join(trial_dir, 'val_loss.csv')
    callbacks.append(ModelCheckpointEveryNBatch(ckpt_path, val_loss_log_path,
                                                     x, yval_bit,
                                                     verbose=1, save_best_only=True,
                                                     period=config['val_check_period']))

    callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0, patience=config['patience'],
                                                   verbose=1, mode='auto'))

    return callbacks


def main():

    if len(sys.argv) != 2:
        print("Invalid number of params! Usage: python train_cvae.py config_path")

    config_path = sys.argv[1]

    config = load_config(config_path)

    root_log_dir = config['root_log_dir']

    if not os.path.exists(root_log_dir):
        os.makedirs(root_log_dir, exist_ok=True)

    experiment_name = '_'.join([config['experiment_name'], get_experiment_id()])

    experiment_dir = os.path.join(root_log_dir, experiment_name)

    os.mkdir(experiment_dir)

    (xtr, ytr_bit, ytr_deg), (xval, yval_bit, yval_deg), (xte, yte_bit, yte_deg) = load_dataset(config['dataset'],
                                                                                                config['data_path'],
                                                                                                config['net_output'])
    eval_data = [xtr, ytr_deg, xval, yval_deg, xte, yte_deg]

    image_height, image_width, n_channels = xtr.shape[1], xtr.shape[2], xtr.shape[3]

    n_trials = config['n_trials']
    best_trial_id = 0

    results = dict()

    if config['random_hyp_search']:
        hyp_params = generate_hyper_params(n_trials)
    else:
        hyp_params = load_hyper_params(config)

    checkpoints = dict()
    results_csv = os.path.join(experiment_dir, 'results.csv')

    for tid in range(0, n_trials):

        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)
        print("logs could be found at %s" % trial_dir)

        print_hyp_params(hyp_params, tid)

        trial_hyp_params = get_trial_hyp_params(hyp_params, tid)

        trial_best_ckpt_path = os.path.join(trial_dir, 'cvae.full_model.trial_%d.best.weights.hdf5' % tid)

        keras_callbacks = define_callbacks(config=config,
                                           trial_dir=trial_dir,
                                           ckpt_path=trial_best_ckpt_path,
                                           val_data=[xval, yval_bit])

        model = CVAE(image_height=image_height,
                     image_width=image_width,
                     n_channels=3,
                     **trial_hyp_params)

        model.save_weights(trial_best_ckpt_path)

        model.fit([xtr, ytr_bit], [xval, yval_bit],
                  batch_size=trial_hyp_params['batch_size'],
                  n_epochs=config['n_epochs'],
                  callbacks=keras_callbacks)

        model.load_weights(trial_best_ckpt_path)

        trial_results = evaluate_model(model, eval_data)

        results[tid] = trial_results
        checkpoints[tid] = trial_best_ckpt_path

        trial_df = results_to_df(trial_results, trial_hyp_params)

        if tid == 0:
            res_df = trial_df
        else:
            res_df = res_df.append(trial_df).reset_index(drop=True)

        res_df.to_csv(results_csv, sep=';')

        if tid > 0:
            if trial_results['validation']['elbo'] > results[best_trial_id]['validation']['elbo']:
                best_trial_id = tid
                print("Better validation loss achieved, current best trial: %d" % best_trial_id)

    print("Loading best model (trial_id = %d)" % best_trial_id)

    best_ckpt_path = checkpoints[best_trial_id]
    overall_best_ckpt_path = os.path.join(experiment_dir, 'cvae.full_model.overall_best.weights.hdf5')
    shutil.copy(best_ckpt_path, overall_best_ckpt_path)

    best_model = CVAE(image_height=image_height,
                      image_width=image_width,
                      n_channels=n_channels,
                      **get_trial_hyp_params(hyp_params, best_trial_id))

    best_model.load_weights(overall_best_ckpt_path)

    print("Evaluating best model..")
    best_results = evaluate_model(best_model, eval_data)

    return


if __name__ == '__main__':
    main()