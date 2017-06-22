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
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np

from scipy.stats import sem

from keras.layers import Input, Dense, Lambda
import keras.backend as K
from keras.models import Model
from keras.models import load_model

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
    custom_objects = {}

    kappa = config['kappa']
    if kappa == 0.0:
        kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(vgg_x))
        kappa_model = Model(X, kappa_pred)
    else:
        kappa_pred = tf.ones([tf.shape(y_pred)[0], 1])*kappa

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

        def _von_mises_neg_log_likelihood_keras(y_true, y_pred):
            return -von_mises_log_likelihood_tf(y_true, y_pred, kappa_pred, input_type='biternion')

        loss_te = _von_mises_neg_log_likelihood_keras
        custom_objects.update({'_von_mises_neg_log_likelihood_keras':_von_mises_neg_log_likelihood_keras})

    else:
        raise ValueError("loss should be 'mad','cosine','von_mises' or 'vm_likelihood'")

    optimizer = get_optimizer(config['optimizer_params'])

    model.compile(loss=loss_te,
                  optimizer=optimizer,
                  metrics=metrics)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=experiment_dir,
                                                       write_images=True)

    train_csv_log = os.path.join(experiment_dir, 'train.csv')
    csv_callback = keras.callbacks.CSVLogger(train_csv_log, separator=',', append=False)

    best_model_weights_file = os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.model.h5')

    model_ckpt_callback = keras.callbacks.ModelCheckpoint(best_model_weights_file,
                                                          save_best_only=True)

    print("logs could be found at %s" % experiment_dir)

    validation_split = config['validation_split']
    if validation_split == 0:
        print("Using test data as validation..")
        model.fit(x=xtr, y=ytr,
                  batch_size=config['batch_size'],
                  epochs=config['n_epochs'],
                  verbose=1,
                  validation_data=(xte, yte),
                  callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])
    else:
        model.fit(x=xtr, y=ytr,
                  batch_size=config['batch_size'],
                  epochs=config['n_epochs'],
                  verbose=1,
                  validation_split=validation_split,
                  callbacks=[tensorboard_callback, csv_callback, model_ckpt_callback])
        # print("fine-tuning the model..")
        # model.optimizer.lr.assign(config['optimizer_params']['learning_rate']*0.01)

    # model.save(os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.h5'))

    # model.load_weights(best_model_weights_file)
    model = load_model(best_model_weights_file,
                       custom_objects=custom_objects)

    results = dict()
    results_yml_file = os.path.join(experiment_dir, 'results.yml')

    if net_output == 'biternion':
        ytr_preds_bit = model.predict(xtr)
        ytr_preds_deg = bit2deg(ytr_preds_bit)
        yte_preds_bit = model.predict(xte)
        yte_preds_deg = bit2deg(yte_preds_bit)

    elif net_output == 'degrees':
        yte_preds_deg = np.squeeze(model.predict(xte))

    if kappa == 0.0:
        print("predicting kappa...")
        kappa_preds_tr = kappa_model.predict(xtr)
        results['mean_kappa_tr'] = float(np.mean(kappa_preds_tr))
        results['std_kappa_tr'] = float(np.std(kappa_preds_tr))
        print("predicted kappa (train) : %f ± %f" % (results['mean_kappa_tr'], results['std_kappa_tr']))
        kappa_preds_te = kappa_model.predict(xte)
        results['mean_kappa_te'] = float(np.mean(kappa_preds_te))
        results['std_kappa_te'] = float(np.std(kappa_preds_te))
        print("predicted kappa (test) : %f ± %f" % (results['mean_kappa_te'], results['std_kappa_te']))
    else:
        # print("fine-tuning kappa as hyper-parameter...")
        # kappa = finetune_kappa(xtr, ytr_bit, model)
        kappa_preds_tr = np.ones(xtr.shape[0]) * kappa
        kappa_preds_te = np.ones(xte.shape[0]) * kappa
        print("kappa value: %f" % kappa)

    loss_tr = maad_from_deg(ytr_preds_deg, ytr_deg)
    results['mean_loss_tr'] = float(np.mean(loss_tr))
    results['std_loss_tr'] = float(np.std(loss_tr))
    print("MAAD error (train) : %f ± %f" % (results['mean_loss_tr'], results['std_loss_tr']))

    loss_te = maad_from_deg(yte_preds_deg, yte_deg)
    results['mean_loss_te'] = float(np.mean(loss_te))
    results['std_loss_te'] = float(np.std(loss_te))
    print("MAAD error (test) : %f ± %f" % (results['mean_loss_te'], results['std_loss_te']))

    if net_output == 'biternion':
        log_likelihoods_tr = von_mises_log_likelihood_np(ytr_bit, ytr_preds_bit, kappa_preds_tr, input_type='biternion')
        results['log_likelihood_mean_tr'] = float(np.mean(log_likelihoods_tr))
        results['log_likelihood_tr_sem'] = float(sem(log_likelihoods_tr, axis=None))
        print("log-likelihood (train) : %f ± %f SEM" % (results['log_likelihood_mean_tr'],
                                                        results['log_likelihood_tr_sem']))
        import ipdb; ipdb.set_trace()

        log_likelihoods_te = von_mises_log_likelihood_np(yte_bit, yte_preds_bit, kappa_preds_te, input_type='biternion')
        results['log_likelihood_mean_te'] = float(np.mean(log_likelihoods_te))
        results['log_likelihood_te_sem'] = float(sem(log_likelihoods_te, axis=None))
        print("log-likelihood (test) : %f ± %f SEM" % (results['log_likelihood_mean_te'],
                                                       results['log_likelihood_te_sem']))

    print("stored model available at %s" % experiment_dir)

    import ipdb; ipdb.set_trace()

    with open(results_yml_file, 'w') as results_yml_file:
        yaml.dump(results, results_yml_file, default_flow_style=False)

    return


if __name__ == '__main__':
    train()