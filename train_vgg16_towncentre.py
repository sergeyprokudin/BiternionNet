import numpy as np
import keras
import os
import yaml

from models import vgg
from utils.angles import deg2bit, bit2deg
from utils.losses import cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.towncentre import load_towncentre


def train():

    config_path = 'train_vgg16_towncentre.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f)

    trained_models_path = config['trained_models_path']
    data_path = config['data_path']
    val_size = config['validation_size']

    xtr, ytr, xte, yte = load_towncentre(data_path, canonical_split=True)
    image_height, image_width = xtr.shape[1], xtr.shape[2]
    ytr_bit = deg2bit(ytr)
    yte_bit = deg2bit(yte)

    model = vgg.vgg_model(n_outputs=2,
                          image_height=image_height,
                          image_width=image_width,
                          l2_normalize_final=True)

    if config['loss'] == 'cosine':
        loss = cosine_loss_tf
    elif config['loss'] == 'von_mises':
        loss = von_mises_loss_tf
    else:
        raise ValueError("loss should be 'cosine' or 'von_mises'")

    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adadelta(rho=.95,
                                                      epsilon=1e-7,
                                                      lr=1),
                  metrics=['cosine'])

    model.fit(x=xtr[val_size:], y=ytr_bit[val_size:],
              batch_size=config['batch_size'],
              epochs=config['n_epochs'],
              verbose=1,
              validation_data=(xtr[0:val_size], ytr_bit[0:val_size]))

    if not os.path.exists(trained_models_path):
        os.mkdir(trained_models_path)
    model.save(os.path.join(trained_models_path, 'vgg_bit_' + config['loss'] + '_town.h5'))

    yte_preds = bit2deg(model.predict(xte))

    loss = maad_from_deg(yte_preds, yte)
    mean_loss = np.mean(loss)
    std_loss = np.std(loss)

    print("MAAD error (test) : %f ± %f" % (mean_loss, std_loss))

    return


if __name__ == '__main__':
    train()