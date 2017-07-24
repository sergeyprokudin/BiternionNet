import keras
import os
import shutil
import yaml
import numpy as np

from models.cvae import CVAE
from utils.angles import deg2bit, bit2deg
from utils.towncentre import load_towncentre
from utils.experiements import get_experiment_id


def main():

    n_u = 8
    exp_id = get_experiment_id()
    root_log_dir = 'logs/cvae/'

    experiment_dir = os.path.join(root_log_dir, exp_id)
    os.mkdir(experiment_dir)

    xtr, ytr_deg, xval, yval_deg, xte, yte_deg = load_towncentre('data/TownCentre.pkl.gz',
                                                                 canonical_split=True,
                                                                 verbose=1)
    ytr_bit = deg2bit(ytr_deg)
    yval_bit = deg2bit(yval_deg)
    yte_bit = deg2bit(yte_deg)

    image_height, image_width, n_channels = xtr.shape[1:]
    phi_shape = yte_bit.shape[1]

    best_trial_id = 0
    n_trials = 1
    results = dict()
    n_epochs = 10

    for tid in range(0, n_trials):

        print("TRIAL %d" % tid)
        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)

        cvae_best_ckpt_path = os.path.join(trial_dir, 'cvae.full_model.trial_%d.best.weights.hdf5' % tid)

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
                          kl_weight=0.6)

        cvae_model.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=50, epochs=n_epochs,
                                  validation_data=([xval, yval_bit], yval_bit),
                                  callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

        cvae_model.evaluate(xtr, ytr_deg, 'train')
        cvae_model.evaluate(xval, yval_deg, 'validation')
        cvae_model.evaluate(xte, yte_deg, 'test')

        kl_weight_range = [0.7, 0.8, 0.9, 1.0]

        for kl_weight in kl_weight_range:

            print('kl weight: %f' % kl_weight)

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
                              kl_weight=kl_weight)

            cvae_model.full_model.load_weights(cvae_best_ckpt_path)

            cvae_model.evaluate(xval, yval_deg, 'validation')
            cvae_model.evaluate(xte, yte_deg, 'test')

            if kl_weight == 1.0:
                n_epochs = 25

            cvae_model.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=50, epochs=n_epochs,
                                      validation_data=([xval, yval_bit], yval_bit),
                                      callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

        # cvae_model = CVAE(image_height=image_height,
        #                   image_width=image_width,
        #                   n_channels=n_channels,
        #                   n_hidden_units=n_u)
        #
        # prior_best_ckpt_path = os.path.join(trial_dir, 'cvae.prior.trial_%d.best.weights.hdf5' % tid)
        #
        # model_ckpt_callback = keras.callbacks.ModelCheckpoint(prior_best_ckpt_path,
        #                                                       monitor='val_loss',
        #                                                       mode='min',
        #                                                       save_best_only=True,
        #                                                       save_weights_only=True,
        #                                                       period=1,
        #                                                       verbose=1)
        #
        # cvae_model.full_model.load_weights(cvae_best_ckpt_path)
        #
        # encoder_preds_tr = cvae_model.encoder_model.predict([xtr, ytr_bit])
        # encoder_preds_val = cvae_model.encoder_model.predict([xval, yval_bit])
        #
        # cvae_model.prior_model.fit([xtr, ytr_bit], [encoder_preds_tr], batch_size=50, epochs=25,
        #                           validation_data=([xval, yval_bit], encoder_preds_val),
        #                           callbacks=[model_ckpt_callback])

        best_model = CVAE(image_height=image_height,
                          image_width=image_width,
                          n_channels=n_channels,
                          n_hidden_units=n_u)

        best_model.full_model.load_weights(cvae_best_ckpt_path)
        # best_model.prior_model.load_weights(prior_best_ckpt_path)

        trial_results = dict()
        trial_results['ckpt_path'] = cvae_best_ckpt_path
        trial_results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
        trial_results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
        trial_results['test'] = best_model.evaluate(xte, yte_deg, 'test')
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
    best_results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
    best_results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
    best_results['test'] = best_model.evaluate(xte, yte_deg, 'test')

    results['best'] = best_results

    results_yml_file = os.path.join(experiment_dir, 'results.yml')
    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)


if __name__ == '__main__':
    main()