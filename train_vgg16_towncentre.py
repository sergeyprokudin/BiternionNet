import numpy as np
import keras
import os


from models import vgg
from utils.angles import deg2bit, bit2deg
from utils.losses import cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.towncentre import load_towncentre


def train():

    xtr, ytr, xte, yte = load_towncentre('data/TownCentre.pkl.gz', canonical_split=True)
    image_height, image_width = xtr.shape[1], xtr.shape[2]
    ytr_bit = deg2bit(ytr)
    yte_bit = deg2bit(yte)

    val_size = 1000

    model = vgg.vgg_model(n_outputs=2,
                            image_height=image_height,
                            image_width=image_width,
                            l2_normalize_final=True)

    model.compile(loss=cosine_loss_tf,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['cosine'])

    model.fit(x=xtr[val_size:], y=ytr_bit[val_size:],
              batch_size=10,
              epochs=10,
              verbose=1,
              validation_data=(xtr[0:val_size], ytr_bit[0:val_size]))

    model.save('trained_models/vgg_bit_town.h5')

    yte_preds = bit2deg(model.predict(xte))

    loss = maad_from_deg(yte_preds, yte)
    mean_loss = np.mean(loss)
    std_loss = np.std(loss)

    print("MAAD error (test) : %f ± %f" % (mean_loss, std_loss))

    # model = load_model('trained_models/vgg_bit_town.h5', custom_objects={'cosine_loss_tf': cosine_loss_tf})

    return


if __name__ == '__main__':
    train()