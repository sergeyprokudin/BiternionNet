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


def vgg_model(n_outputs=1, final_layer=False, l2_normalize_final=False,
              image_height=50, image_width=50,
              conv_dropout_val=0.2, fc_dropout_val=0.5, fc_layer_size=512):

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
    model.add(Dropout(conv_dropout_val))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(fc_dropout_val))

    if final_layer:
        model.add(Dense(n_outputs, activation=None))
        if l2_normalize_final:
            model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    return model


class DegreeVGG:

    def __init__(self,
             image_height=50,
             image_width=50,
             n_channels=3,
             n_outputs=1,
             predict_kappa=False,
             fixed_kappa_value=1.0):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels

        self.X = Input(shape=[image_height, image_width, 3])

        self.vgg_x = vgg_model(n_outputs=1,
                          final_layer=True,
                          image_height=self.image_height,
                          image_width=self.image_width)(self.X)

        self.model = Model(self.X, self.vgg_x)

    def evaluate(self, x, ytrue_deg, data_part):

        ypreds_deg = np.squeeze(self.model.predict(x))

        loss = maad_from_deg(ypreds_deg, ytrue_deg)

        results = dict()

        results['maad_loss'] = float(np.mean(loss))
        results['maad_loss_sem'] = float(sem(loss, axis=None))
        print("MAAD error (%s) : %f ± %fSEM" % (data_part,
                                             results['maad_loss'],
                                             results['maad_loss_sem']))

        return results


class BiternionVGG:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 predict_kappa=False,
                 fixed_kappa_value=1.0,
                 conv_dropout_val=0.2,
                 fc_dropout_val=0.5):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.predict_kappa = predict_kappa
        self.fixed_kappa_value = fixed_kappa_value
        self.conv_dropout_val = conv_dropout_val
        self.fc_dropout_val = fc_dropout_val

        self.X = Input(shape=[image_height, image_width, 3])

        vgg_x = vgg_model(final_layer=False,
                          image_height=self.image_height,
                          image_width=self.image_width,
                          conv_dropout_val=self.conv_dropout_val,
                          fc_dropout_val=self.fc_dropout_val)(self.X)

        self.y_pred = Lambda(lambda x: K.l2_normalize(x, axis=1))(Dense(2)(vgg_x))

        if self.predict_kappa:
            self.kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(vgg_x))
            self.model = Model(self.X, concatenate([self.y_pred, self.kappa_pred]))
        else:
            self.model = Model(self.X, self.y_pred)

    def evaluate(self, x, ytrue_deg, data_part):

        ytrue_bit = deg2bit(ytrue_deg)
        ypreds = self.model.predict(x)
        ypreds_bit = ypreds[:, 0:2]
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

