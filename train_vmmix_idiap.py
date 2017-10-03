import keras
import os
import sys
import shutil
import yaml
import numpy as np
import pandas as pd
import itertools

from models.vgg_vmmix import BiternionVGGMixture
from utils.load_datasets import load_dataset
from utils.experiements import get_experiment_id


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

    image_height, image_width, n_channels = xtr.shape[1:]

    best_trial_id = 0

    results = dict()

    n_epochs = config['n_epochs']
    n_trials = config['n_trials']
    batch_sizes = config['batch_sizes']
    learning_rates = config['learning_rates']
    n_components_lst = config['n_components']
    params_grid = list(itertools.product(learning_rates, batch_sizes, n_components_lst))*n_trials

    res_cols = ['trial_id', 'batch_size', 'learning_rate',  'n_components',
                'val_maad', 'val_likelihood', 'test_maad', 'test_likelihood']

    results_df = pd.DataFrame(columns=res_cols)
    results_csv = os.path.join(experiment_dir, 'results.csv')

    for tid, params in enumerate(params_grid):

        learning_rate = params[0]
        batch_size = params[1]
        n_components = params[2]

        print("TRIAL %d // %d" % (tid, len(params_grid)))
        print("batch_size: %d" % batch_size)
        print("learning_rate: %f" % learning_rate)
        print("n_components: %f" % n_components)

        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)

        vmmix_best_ckpt_path = os.path.join(trial_dir, 'vmmix.full_model.trial_%d.best.weights.hdf5' % tid)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=trial_dir)

        train_csv_log = os.path.join(trial_dir, 'train.csv')
        csv_callback = keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False)

        model_ckpt_callback = keras.callbacks.ModelCheckpoint(vmmix_best_ckpt_path,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              period=1,
                                                              verbose=1)

        vggmix_model = BiternionVGGMixture(image_height=image_height,
                                           image_width=image_width,
                                           n_channels=n_channels,
                                           n_components=n_components,
                                           learning_rate=learning_rate)

        vggmix_model.model.save_weights(vmmix_best_ckpt_path)

        vggmix_model.model.fit(xtr, ytr_bit, batch_size=batch_size, epochs=n_epochs,
                               validation_data=(xval, yval_bit),
                               callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

        best_model = BiternionVGGMixture(image_height=image_height,
                                         image_width=image_width,
                                         n_channels=n_channels,
                                         n_components=n_components)

        best_model.model.load_weights(vmmix_best_ckpt_path)

        trial_results = dict()
        trial_results['ckpt_path'] = vmmix_best_ckpt_path
        trial_results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
        trial_results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
        trial_results['test'] = best_model.evaluate(xte, yte_deg, 'test')
        results[tid] = trial_results

        results_np = np.asarray([tid, batch_size, learning_rate, n_components,
                                 trial_results['validation']['maad_loss'],
                                 trial_results['validation']['log_likelihood_mean'],
                                 trial_results['test']['maad_loss'],
                                 trial_results['test']['log_likelihood_mean']]).reshape([1, 8])

        trial_res_df = pd.DataFrame(results_np, columns=res_cols)
        results_df = results_df.append(trial_res_df)
        results_df.to_csv(results_csv)

        if tid > 0:
            if trial_results['validation']['log_likelihood_mean'] > \
                    results[best_trial_id]['validation']['log_likelihood_mean']:
                best_trial_id = tid
                print("Better validation loss achieved, current best trial: %d" % best_trial_id)

    print("Loading best model (trial_id = %d)" % best_trial_id)

    best_ckpt_path = results[best_trial_id]['ckpt_path']
    overall_best_ckpt_path = os.path.join(experiment_dir, 'vmmix.full_model.overall_best.weights.hdf5')
    shutil.copy(best_ckpt_path, overall_best_ckpt_path)

    best_model_n_components = params_grid[best_trial_id][2]

    best_model = BiternionVGGMixture(image_height=image_height,
                                     image_width=image_width,
                                     n_channels=n_channels,
                                     n_components=best_model_n_components)
    best_model.model.load_weights(overall_best_ckpt_path)

    best_results = dict()
    best_results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
    best_results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
    best_results['test'] = best_model.evaluate(xte, yte_deg, 'test')

    results['best'] = best_results

    results_yml_file = os.path.join(experiment_dir, 'results.yml')
    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)

    results_df.to_csv(results_csv)


if __name__ == '__main__':
    main()