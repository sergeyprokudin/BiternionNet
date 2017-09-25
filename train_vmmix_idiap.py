import keras
import os
import shutil
import yaml
import numpy as np
import pandas as pd
import itertools

from models.vgg_vmmix import BiternionVGGMixture
from utils.angles import deg2bit, rad2bit, bit2deg
from utils.idiap import load_idiap
from utils.experiements import get_experiment_id
from utils.hyper_tune import make_lr_batch_size_grid


def main():

    exp_id = get_experiment_id()

    root_log_dir = 'logs/IDIAP/pan/vmmix'

    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)

    experiment_dir = os.path.join(root_log_dir, exp_id)
    os.mkdir(experiment_dir)

    (xtr, ptr_rad, ttr_rad, rtr_rad, names_tr), \
    (xval, pval_rad, tval_rad, rval_rad, names_val), \
    (xte, pte_rad, tte_rad, rte_rad, names_te) = load_idiap('data//IDIAP.pkl')

    image_height, image_width = xtr.shape[1], xtr.shape[2]

    net_output = 'pan'

    if net_output == 'pan':
        ytr = rad2bit(ptr_rad)
        yval = rad2bit(pval_rad)
        yte = rad2bit(pte_rad)
        ytr_deg = np.rad2deg(ptr_rad)
        yval_deg = np.rad2deg(pval_rad)
        yte_deg = np.rad2deg(pte_rad)
    elif net_output == 'tilt':
        ytr = rad2bit(ttr_rad)
        yval = rad2bit(tval_rad)
        yte = rad2bit(tte_rad)
        ytr_deg = np.rad2deg(ttr_rad)
        yval_deg = np.rad2deg(tval_rad)
        yte_deg = np.rad2deg(tte_rad)
    elif net_output == 'roll':
        ytr = rad2bit(rtr_rad)
        yval = rad2bit(rval_rad)
        yte = rad2bit(rte_rad)
        ytr_deg = np.rad2deg(rtr_rad)
        yval_deg = np.rad2deg(rval_rad)
        yte_deg = np.rad2deg(rte_rad)
    else:
        raise ValueError("net_output should be 'pan', 'tilt' or 'roll'")

    # xtr, ytr_deg = aug_data(xtr, ytr_deg)
    # xval, yval_deg = aug_data(xval, yval_deg)
    # xte, yte_deg = aug_data(xval, yval_deg)

    ytr_bit = deg2bit(ytr_deg)
    yval_bit = deg2bit(yval_deg)
    yte_bit = deg2bit(yte_deg)

    image_height, image_width, n_channels = xtr.shape[1:]
    phi_shape = yte_bit.shape[1]

    best_trial_id = 0
    n_trials = 4
    results = dict()

    # best so far
    n_epochs = 50
    # batch_size = 64
    n_components = 5
    # learning_rate = 1.0e-6

    batch_sizes = [64, 128, 256]
    learning_rates = [1.0e-6, 1.0e-7, 1.0e-5]
    n_components = [5, 10, 3]
    params_grid = list(itertools.product(learning_rates, batch_sizes))*n_trials

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
                                 trial_results['test']['log_likelihood_mean']]).reshape([1, 7])

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

    best_model = BiternionVGGMixture(image_height=image_height,
                                     image_width=image_width,
                                     n_channels=n_channels,
                                     n_components=n_components)
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