import numpy as np
import keras
import os
import yaml
import binascii
import shutil

from models import vgg
from utils.angles import deg2bit, bit2deg
from utils.losses import cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.towncentre import load_towncentre


def get_exp_id():
    experiment_id = binascii.hexlify(os.urandom(10))
    # experiment_name = '_'.join([config['']])
    return experiment_id

def train():

    config_path = 'train_vgg16_towncentre.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    root_log_dir = config['root_log_dir']
    data_path = config['data_path']

    if not os.path.exists(root_log_dir):
        os.mkdir(root_log_dir)
    experiment_id = get_exp_id()
    experiment_dir = os.path.join(root_log_dir, str(experiment_id))
    os.mkdir(experiment_dir)
    shutil.copy(config_path, experiment_dir)

    xtr, ytr, xte, yte = load_towncentre(data_path, canonical_split=True)
    image_height, image_width = xtr.shape[1], xtr.shape[2]
    ytr_bit = deg2bit(ytr)
    yte_bit = deg2bit(yte)

    if config['loss'] == 'cosine':
        loss = cosine_loss_tf
    elif config['loss'] == 'von_mises':
        loss = von_mises_loss_tf
    else:
        raise ValueError("loss should be 'cosine' or 'von_mises'")

    model = vgg.vgg_model(n_outputs=2,
                          image_height=image_height,
                          image_width=image_width,
                          l2_normalize_final=True)

    optimizer_params = config['optimizer_params']

    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adadelta(rho=optimizer_params['rho'],
                                                      epsilon=optimizer_params['epsilon'],
                                                      lr=optimizer_params['learning_rate']),
                  metrics=['cosine'])

    val_size = config['validation_size']
    model.fit(x=xtr[val_size:], y=ytr_bit[val_size:],
              batch_size=config['batch_size'],
              epochs=config['n_epochs'],
              verbose=1,
              validation_data=(xtr[0:val_size], ytr_bit[0:val_size]))

    model.save(os.path.join(experiment_dir, 'vgg_bit_' + config['loss'] + '_town.h5'))

    yte_preds = bit2deg(model.predict(xte))

    loss = maad_from_deg(yte_preds, yte)
    mean_loss = np.mean(loss)
    std_loss = np.std(loss)

    print("MAAD error (test) : %f ± %f" % (mean_loss, std_loss))

    return


if __name__ == '__main__':
    train()