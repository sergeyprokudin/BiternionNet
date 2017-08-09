import tensorflow as tf
import keras
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate

from utils.angles import deg2bit, bit2deg
from utils.losses import maad_from_deg, von_mises_log_likelihood_np
from scipy.stats import sem

N_BITERNION_OUTPUT = 2


def vgg_model(n_outputs=1, final_layer=False, l2_normalize_final=False,
              image_height=50, image_width=50):
    model = Sequential()

    model.add(Conv2D(24, kernel_size=(3, 3),
                     activation=None,
                     input_shape=[image_height, image_width, 3]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    if final_layer:
        model.add(Dense(n_outputs, activation=None))
        if l2_normalize_final:
            model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    return model


class BiternionVGGMixture:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 n_components=5,
                 hlayer_size=256):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.n_components = n_components
        self.hlayer_size = hlayer_size

        self.X = Input(shape=[image_height, image_width, 3])

        vgg_x = vgg_model(final_layer=False,
                          image_height=self.image_height,
                          image_width=self.image_width)(self.X)

        mu_preds = []
        for i in range(0, self.n_components):
            mu_pred = Dense(N_BITERNION_OUTPUT)(Dense(self.hlayer_size)(vgg_x))
            mu_pred_normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(mu_pred)
            mu_pred_norm_reshaped = K.reshape(mu_pred_normalized, [-1, 1, N_BITERNION_OUTPUT])
            mu_preds.append(mu_pred_norm_reshaped)
        self.mu_preds = concatenate(mu_preds, axis=1)

        self.kappa_preds = Lambda(lambda x: K.abs(x))(Dense(self.n_components)(Dense(256)(vgg_x)))
        #self.kappa_preds = K.reshape(kappa_preds, [-1, self.n_components, 1])

        self.component_probs = Lambda(lambda x: K.softmax(x))(Dense(self.n_components)(Dense(256)(vgg_x)))
        #self.component_probs = K.reshape(component_probs, [-1, self.n_components, 1])

        self.y_pred = concatenate([self.kappa_preds, self.component_probs])

        self.model = Model(self.X, self.y_pred)

    def parse(self, y_preds):
        return

    def evaluate(self, x, ytrue_deg, data_part):

        ytrue_bit = deg2bit(ytrue_deg)
        ypreds = self.model.predict(x)
        ypreds_bit = ypreds[:, 0:N_BITERNION_OUTPUT]
        ypreds_deg = bit2deg(ypreds_bit)

        if self.predict_kappa:
            kappa_preds = ypreds[:, 2:]
        else:
            kappa_preds = np.ones([ytrue_deg.shape[0], 1]) * self.fixed_kappa_value

        loss = maad_from_deg(ypreds_deg, ytrue_deg)

        results = dict()

        results['maad_loss'] = float(np.mean(loss))
        results['maad_loss_sem'] = float(sem(loss))
        print("MAAD error (%s) : %f ± %fSEM" % (data_part,
                                             results['maad_loss'],
                                             results['maad_loss_sem']))

        results['mean_kappa'] = float(np.mean(kappa_preds))
        results['std_kappa'] = float(np.std(kappa_preds))

        log_likelihoods = von_mises_log_likelihood_np(ytrue_bit, ypreds_bit, kappa_preds)

        results['log_likelihood_mean'] = float(np.mean(log_likelihoods))
        results['log_likelihood_sem'] = float(sem(log_likelihoods, axis=None))
        print("log-likelihood (%s) : %f ± %fSEM" % (data_part,
                                                    results['log_likelihood_mean'],
                                                    results['log_likelihood_sem']))

        return results

