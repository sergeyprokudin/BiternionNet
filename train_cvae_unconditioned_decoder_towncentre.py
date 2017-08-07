import keras
import os
import shutil
import yaml
import numpy as np

from models.cvae_unconditioned_decoder import CVAE
from utils.angles import deg2bit, bit2deg
from utils.towncentre import load_towncentre, aug_data
from utils.experiements import get_experiment_id
from utils.custom_keras_callbacks import EvalCVAEModel


def main():

    n_u = 128
    exp_id = get_experiment_id()
    root_log_dir = 'logs/cvae/'

    experiment_dir = os.path.join(root_log_dir, exp_id)
    os.mkdir(experiment_dir)

    xtr, ytr_deg, xval, yval_deg, xte, yte_deg = load_towncentre('data/TownCentre.pkl.gz',
                                                                 canonical_split=True,
                                                                 verbose=1)

    # xtr, ytr_deg = aug_data(xtr, ytr_deg)
    # xval, yval_deg = aug_data(xval, yval_deg)
    # xte, yte_deg = aug_data(xval, yval_deg)

    ytr_bit = deg2bit(ytr_deg)
    yval_bit = deg2bit(yval_deg)
    yte_bit = deg2bit(yte_deg)

    image_height, image_width, n_channels = xtr.shape[1:]
    phi_shape = yte_bit.shape[1]

    best_trial_id = 0
    n_trials = 5
    results = dict()

    n_epochs = 100
    batch_size = 10

    for tid in range(0, n_trials):

        print("TRIAL %d" % tid)
        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)

        cvae_best_ckpt_path = os.path.join(trial_dir, 'cvae.full_model.trial_%d.best.weights.hdf5' % tid)
        cvae_bestloglike_ckpt_path = os.path.join(trial_dir, 'cvae.full_model.trial_%d.best_likelihood.weights.hdf5'
                                                  % tid)

        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=trial_dir)

        train_csv_log = os.path.join(trial_dir, 'train.csv')
        csv_callback = keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False)

        model_ckpt_callback = keras.callbacks.ModelCheckpoint(cvae_best_ckpt_path,
                                                              monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=True,
                                                              save_weights_only=True,
                                                              period=1,
                                                              verbose=1)

        cvae_model = CVAE(image_height=image_height,
                          image_width=image_width,
                          n_channels=n_channels,
                          n_hidden_units=n_u,
                          kl_weight=0.1)

        eval_callback = EvalCVAEModel(xval, yval_deg, 'validation', cvae_model, cvae_bestloglike_ckpt_path)

        cvae_model.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=batch_size, epochs=n_epochs,
                                  validation_data=([xval, yval_bit], yval_bit),
                                  callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback, eval_callback])

        best_model = CVAE(image_height=image_height,
                          image_width=image_width,
                          n_channels=n_channels,
                          n_hidden_units=n_u)

        best_model.full_model.load_weights(cvae_best_ckpt_path)

        trial_results = dict()
        trial_results['ckpt_path'] = cvae_best_ckpt_path
        trial_results['train'] = best_model.evaluate_multi(xtr, ytr_deg, 'train')
        trial_results['validation'] = best_model.evaluate_multi(xval, yval_deg, 'validation')
        trial_results['test'] = best_model.evaluate_multi(xte, yte_deg, 'test')
        results[tid] = trial_results

        if tid > 0:
            if trial_results['validation']['elbo'] > results[best_trial_id]['validation']['elbo']:
                best_trial_id = tid
                print("Better validation loss achieved, current best trial: %d" % best_trial_id)

    print("Loading best model (trial_id = %d)" % best_trial_id)

    best_ckpt_path = results[best_trial_id]['ckpt_path']
    overall_best_ckpt_path = os.path.join(experiment_dir, 'cvae.full_model.overall_best.weights.hdf5')
    shutil.copy(best_ckpt_path, overall_best_ckpt_path)

    best_model = CVAE(image_height=image_height,
                      image_width=image_width,
                      n_channels=n_channels,
                      n_hidden_units=n_u)
    best_model.full_model.load_weights(overall_best_ckpt_path)

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