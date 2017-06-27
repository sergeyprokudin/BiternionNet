from keras.layers import Input, Dense, Lambda, Flatten, Activation, Merge, Concatenate, Add
from keras import layers
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import numpy as np
import keras.backend as K
import tensorflow as tf

from models import vgg
from utils.losses  import von_mises_log_likelihood_tf


class CVAE:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 n_hidden_units=8):

        self.n_u = n_hidden_units
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.phi_shape = 2

        self.x = Input(shape=[self.image_height, self.image_width, self.n_channels])
        self.phi = Input(shape=[self.phi_shape])
        self.u = Input(shape=[self.n_u])

        self.x_vgg = vgg.vgg_model(image_height=self.image_height,
                                   image_width=self.image_width)(self.x)

        self.x_vgg_shape = self.x_vgg.get_shape().as_list()[1]

        self.mu_encoder, self.log_sigma_encoder = self._encoder_mu_log_sigma()

        self.u_encoder = Lambda(self._sample_u)([self.mu_encoder, self.log_sigma_encoder])

        self.x_vgg_u = concatenate([self.x_vgg, self.u_encoder])

        self.decoder_mu_seq, self.decoder_kappa_seq = self._decoder_net_seq()

        self.full_model = Model(inputs=[self.x, self.phi],
                                outputs=concatenate([self.mu_encoder,
                                                     self.log_sigma_encoder,
                                                     self.decoder_mu_seq(self.x_vgg_u),
                                                     self.decoder_kappa_seq(self.x_vgg_u)]))

        self.full_model.compile()

        self.decoder_input = concatenate([self.x_vgg, self.u])
        self.decoder_model = Model(inputs=[self.x, self.u],
                                   outputs=concatenate([self.decoder_mu_seq(self.decoder_input),
                                                        self.decoder_kappa_seq(self.decoder_input)]))

    def _encoder_mu_log_sigma(self):

        x_vgg_phi = concatenate([self.x_vgg, self.phi])

        hidden = Dense(512, activation='relu')(x_vgg_phi)

        mu_encoder = Dense(self.n_u, activation='linear')(hidden)
        log_sigma_encoder = Dense(self.n_u, activation='linear')(hidden)

        return mu_encoder, log_sigma_encoder

    def _sample_u(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=[self.n_u], mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps

    def _decoder_net_seq(self):
        decoder_mu = Sequential()
        decoder_mu.add(Dense(512, activation='relu',input_shape=[self.x_vgg_shape + self.n_u]))
        # decoder_mu.add(Dense(512, activation='relu'))
        decoder_mu.add(Dense(2, activation='linear'))
        decoder_mu.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

        decoder_kappa = Sequential()
        decoder_kappa.add(Dense(512, activation='relu', input_shape=[self.x_vgg_shape + self.n_u]))
        # decoder_kappa.add(Dense(512, activation='relu'))
        decoder_kappa.add(Dense(1, activation='linear'))

        return decoder_mu, decoder_kappa