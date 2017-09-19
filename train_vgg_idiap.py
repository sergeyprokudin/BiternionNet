import numpy as np
import keras
import tensorflow as tf
import os
import sys
import yaml
import shutil

from models import vgg
from utils.angles import rad2bit
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf
from utils.idiap import load_idiap
from utils.experiements import get_experiment_id
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np, von_mises_neg_log_likelihood_keras

from keras import backend as K


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

    (xtr, ptr_rad, ttr_rad, rtr_rad, names_tr), \
    (xval, pval_rad, tval_rad, rval_rad, names_val), \
    (xte, pte_rad, tte_rad, rte_rad, names_te) = load_idiap('data//IDIAP.pkl')

    image_height, image_width = xtr.shape[1], xtr.shape[2]

    ptr_bit = rad2bit(ptr_rad)
    pval_bit = rad2bit(pval_rad)
    pte_bit = rad2bit(pte_rad)

    ptr_deg = np.rad2deg(ptr_rad)
    pval_deg = np.rad2deg(pval_rad)
    pte_deg = np.rad2deg(pte_rad)

    net_output = config['net_output']
    if net_output == 'biternion':
        ytr = ptr_bit
        yval = pval_bit
        yte = pte_bit
    elif net_output == 'degrees':
        ytr = ptr_deg
        yval = pval_deg
        yte = pte_deg
    else:
        raise ValueError("net_output should be 'biternion' or 'degrees'")

    import ipdb; ipdb.set_trace()

    net_output = config['net_output']

    predict_kappa = config['predict_kappa']
    fixed_kappa_value = config['fixed_kappa_value']

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
        if predict_kappa:
            loss_te = von_mises_neg_log_likelihood_keras
        else:

            def _von_mises_neg_log_likelihood_keras_fixed(y_true, y_pred):
                mu_pred = y_pred[:, 0:2]
                kappa_pred = tf.ones([tf.shape(y_pred[:, 2:])[0], 1])*fixed_kappa_value
                return -K.mean(von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred, net_output))

            loss_te = _von_mises_neg_log_likelihood_keras_fixed
    else:
        raise ValueError("loss should be 'mad','cosine','von_mises' or 'vm_likelihood'")

    optimizer = get_optimizer(config['optimizer_params'])

    best_trial_id = 0
    n_trials = 5
    results = dict()

    for tid in range(0, n_trials):

        print("TRIAL %d" % tid)
        trial_dir = os.path.join(experiment_dir, str(tid))
        os.mkdir(trial_dir)
        print("logs could be found at %s" % trial_dir)

        vgg_model = vgg.BiternionVGG(image_height=image_height,
                                     image_width=image_width,
                                     n_channels=3,
                                     predict_kappa=predict_kappa,
                                     fixed_kappa_value=fixed_kappa_value)

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

        vgg_model.model.fit(x=xtr, y=ytr,
                            batch_size=config['batch_size'],
                            epochs=config['n_epochs'],
                            verbose=1,
                            validation_data=(xval, yval),
                            callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])

        # final_model_ckpt_file = os.path.join(trial_dir, 'vgg_bit_' + config['loss'] + '_town.final.weigths.h5')
        # vgg_model.model.save_weights(final_model_ckpt_file)

        best_model = vgg.BiternionVGG(image_height=image_height,
                                      image_width=image_width,
                                      n_channels=3,
                                      predict_kappa=predict_kappa,
                                      fixed_kappa_value=fixed_kappa_value)

        best_model.model.load_weights(best_model_weights_file)

        trial_results = dict()
        trial_results['ckpt_path'] = best_model_weights_file
        trial_results['train'] = best_model.evaluate(xtr, ytr, 'train')
        trial_results['validation'] = best_model.evaluate(xval, yval, 'validation')
        trial_results['test'] = best_model.evaluate(xte, yte, 'test')
        results[tid] = trial_results

        if tid > 0:
            if trial_results['validation']['log_likelihood_mean'] > \
                    results[best_trial_id]['validation']['log_likelihood_mean']:
                best_trial_id = tid
                print("Better validation loss achieved, current best trial: %d" % best_trial_id)

    print("evaluating model..")
    best_ckpt_path = results[best_trial_id]['ckpt_path']
    overall_best_ckpt_path = os.path.join(experiment_dir, 'vgg.full_model.overall_best.weights.hdf5')
    shutil.copy(best_ckpt_path, overall_best_ckpt_path)

    best_model = vgg.BiternionVGG(image_height=image_height,
                                  image_width=image_width,
                                  n_channels=3,
                                  predict_kappa=predict_kappa,
                                  fixed_kappa_value=fixed_kappa_value)

    best_model.model.load_weights(overall_best_ckpt_path)

    best_results = dict()
    best_results['train'] = best_model.evaluate(xtr, ytr, 'train')
    best_results['validation'] = best_model.evaluate(xval, yval, 'validation')
    best_results['test'] = best_model.evaluate(xte, yte, 'test')
    results['best'] = best_results

    results_yml_file = os.path.join(experiment_dir, 'results.yml')
    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)

    return


if __name__ == '__main__':
    train()