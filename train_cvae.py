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
from utils.custom_keras_callbacks import EvalCVAEModel
from utils import hyper_tune as ht


def load_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    return config


def main():

    if len(sys.argv) != 2:
        print("Invalid number of params! Usage: python train_vgg.py config_path")

    config_path = sys.argv[1]

    config = load_config(config_path)

    root_log_dir = config['root_log_dir']

    if not os.path.exists(root_log_dir):
        os.makedirs(root_log_dir, exist_ok=True)

    experiment_name = '_'.join([config['experiment_name'], get_experiment_id()])

    experiment_dir = os.path.join(root_log_dir, experiment_name)

    os.mkdir(experiment_dir)

    (xtr, ytr_bit, ytr_deg), (xval, yval_bit, yval_deg), (xte, yte_bit, yte_deg) = load_dataset(config)

    image_height, image_width, n_channels = xtr.shape[1], xtr.shape[2], xtr.shape[3]

    n_trials = config['n_trials']
    best_trial_id = 0

    results = dict()

    if not config['random_hyp_search']:

        batch_sizes = config['batch_sizes']
        learning_rates = config['learning_rates']
        params_grid = np.asarray(list(itertools.product(learning_rates, batch_sizes))*n_trials)
        learning_rates = params_grid[:, 0]
        batch_sizes = params_grid[:, 1].astype('int')
        lr_decays = np.ones(n_trials)*0.0
        epsilons = np.ones(n_trials)*1.0e-7
        conv_dropouts = np.ones(n_trials)*config['conv_dropout']
        fc_dropouts = np.ones(n_trials)*config['fc_dropout']
        n_hidden_units_lst = np.ones(n_trials, dtype=int)*config['n_hidden_units']

    else:
        learning_rates = ht.sample_exp_float(n_trials, base=10, min_factor=-7, max_factor=0)
        batch_sizes = ht.sample_exp_int(n_trials, base=2, min_factor=1, max_factor=10)
        lr_decays = np.ones(n_trials)*0.0
        epsilons = np.ones(n_trials)*1.0e-7
        conv_dropouts = np.ones(n_trials)*config['conv_dropout']
        fc_dropouts = np.ones(n_trials)*config['fc_dropout']
        n_hidden_units_lst = np.ones(n_trials, dtype=int)*config['n_hidden_units']

    res_cols = ['trial_id', 'batch_size', 'learning_rate',  'n_hidden_units',
                'val_maad', 'val_elbo', 'val_importance_likelihood',
                'test_maad', 'test_likelihood', 'te_importance_likelihood']

    results_df = pd.DataFrame(columns=res_cols)
    results_csv = os.path.join(experiment_dir, 'results.csv')

    for tid in range(0, n_trials):

        learning_rate = learning_rates[tid]
        batch_size = batch_sizes[tid]
        lr_decay = lr_decays[tid]
        epsilon = epsilons[tid]
        fc_dropout = fc_dropouts[tid]
        conv_dropout = conv_dropouts[tid]
        n_hidden_units = n_hidden_units_lst[tid]

        print("TRIAL %d // %d" % (tid, n_trials))
        print("batch_size: %d" % batch_size)
        print("learning_rate: %f" % learning_rate)
        print("weight decay: %f" % lr_decay)
        print("epsilons: %f" % epsilon)
        print("conv dropout value: %f" % conv_dropout)
        print("fc dropout value: %f" % fc_dropout)
        print("n_hidden_units: %f" % n_hidden_units)

        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)
        print("logs could be found at %s" % trial_dir)

        cvae_best_ckpt_path = os.path.join(trial_dir, 'cvae.full_model.trial_%d.best.weights.hdf5' % tid)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=trial_dir)

        train_csv_log = os.path.join(trial_dir, 'train.csv')
        csv_callback = keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False)

        # model_ckpt_callback = keras.callbacks.ModelCheckpoint(cvae_best_ckpt_path,
        #                                                       monitor='val_loss',
        #                                                       mode='min',
        #                                                       save_best_only=True,
        #                                                       save_weights_only=True,
        #                                                       period=1,
        #                                                       verbose=1)

        val_loss_log_path = os.path.join(trial_dir, 'val_loss.csv')

        model_ckpt_callback = ModelCheckpointEveryNBatch(cvae_best_ckpt_path, val_loss_log_path,
                                                         [xval, yval_bit], yval_bit,
                                                         verbose=1, save_best_only=True,
                                                         period=config['val_check_period'])

        cvae_model = CVAE(image_height=image_height,
                          image_width=image_width,
                          n_channels=3,
                          n_hidden_units=n_hidden_units,
                          learning_rate=learning_rate)

        cvae_model.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=batch_size, epochs=config['n_epochs'],
                                  validation_data=([xval, yval_bit], yval_bit),
                                  callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

        best_model = CVAE(image_height=image_height,
                          image_width=image_width,
                          n_channels=n_channels,
                          n_hidden_units=n_hidden_units)

        best_model.full_model.load_weights(cvae_best_ckpt_path)

        trial_results = dict()
        trial_results['ckpt_path'] = cvae_best_ckpt_path
        trial_results['train'] = best_model.evaluate_multi(xtr, ytr_deg, 'train')
        trial_results['validation'] = best_model.evaluate_multi(xval, yval_deg, 'validation')
        trial_results['test'] = best_model.evaluate_multi(xte, yte_deg, 'test')
        results[tid] = trial_results

        results_np = np.asarray([tid, batch_size, learning_rate, n_hidden_units,
                                 trial_results['validation']['maad_loss'],
                                 trial_results['validation']['elbo'],
                                 trial_results['validation']['importance_log_likelihood'],
                                 trial_results['test']['maad_loss'],
                                 trial_results['test']['elbo'],
                                 trial_results['test']['importance_log_likelihood']]).reshape([1, 10])

        trial_res_df = pd.DataFrame(results_np, columns=res_cols)
        results_df = results_df.append(trial_res_df)
        results_df.to_csv(results_csv)

        if tid > 0:
            if trial_results['validation']['elbo'] > results[best_trial_id]['validation']['elbo']:
                best_trial_id = tid
                print("Better validation loss achieved, current best trial: %d" % best_trial_id)

    print("Loading best model (trial_id = %d)" % best_trial_id)

    best_ckpt_path = results[best_trial_id]['ckpt_path']
    overall_best_ckpt_path = os.path.join(experiment_dir, 'cvae.full_model.overall_best.weights.hdf5')
    shutil.copy(best_ckpt_path, overall_best_ckpt_path)

    best_model_n_hidden_units = params_grid[best_trial_id][2]

    best_model = CVAE(image_height=image_height,
                      image_width=image_width,
                      n_channels=n_channels,
                      n_hidden_units=best_model_n_hidden_units)

    best_model.full_model.load_weights(overall_best_ckpt_path)

    print("Evaluating best model..")
    best_results = dict()
    best_results['train'] = best_model.evaluate_multi(xtr, ytr_deg, 'train')
    best_results['validation'] = best_model.evaluate_multi(xval, yval_deg, 'validation')
    best_results['test'] = best_model.evaluate_multi(xte, yte_deg, 'test')

    results['best'] = best_results

    results_yml_file = os.path.join(experiment_dir, 'results.yml')
    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)


if __name__ == '__main__':
    main()