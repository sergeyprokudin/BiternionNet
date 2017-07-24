import numpy as np

from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model, Sequential

import keras.backend as K

from models import vgg

from utils.losses import gaussian_kl_divergence_tf, gaussian_kl_divergence_np
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np
from utils.angles import deg2bit, bit2deg
from utils.losses import maad_from_deg
from scipy.stats import sem


class CVAE:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 n_hidden_units=8,
                 kl_weight=1.0,
                 rec_weight=1.0):

        self.n_u = n_hidden_units
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.phi_shape = 2
        self.kl_weight = kl_weight
        self.rec_weight = rec_weight

        self.x = Input(shape=[self.image_height, self.image_width, self.n_channels])
        self.phi = Input(shape=[self.phi_shape])
        self.u = Input(shape=[self.n_u])

        self.x_vgg_encoder = vgg.vgg_model(image_height=self.image_height,
                                           image_width=self.image_width)(self.x)

        self.x_vgg_prior = vgg.vgg_model(image_height=self.image_height,
                                         image_width=self.image_width)(self.x)

        self.x_vgg_decoder = vgg.vgg_model(image_height=self.image_height,
                                           image_width=self.image_width)(self.x)

        self.x_vgg_shape = self.x_vgg_encoder.get_shape().as_list()[1]

        self.mu_encoder, self.log_sigma_encoder = self._encoder_mu_log_sigma()

        self.mu_prior, self.log_sigma_prior = self._prior_mu_log_sigma()

        self.u_prior = Lambda(self._sample_u)([self.mu_prior, self.log_sigma_prior])
        # self.u_prior = Lambda(self._sample_normal)([self.mu_prior, self.log_sigma_prior])
        self.u_encoder = Lambda(self._sample_u)([self.mu_encoder, self.log_sigma_encoder])

        self.x_vgg_u = concatenate([self.x_vgg_decoder, self.u_encoder])

        self.decoder_mu_seq, self.decoder_kappa_seq = self._decoder_net_seq()

        self.full_model = Model(inputs=[self.x, self.phi],
                                outputs=concatenate([self.mu_prior,
                                                     self.log_sigma_prior,
                                                     self.mu_encoder,
                                                     self.log_sigma_encoder,
                                                     # self.decoder_mu_seq(self.u_encoder),
                                                     # self.decoder_kappa_seq(self.u_encoder)]))
                                                     self.decoder_mu_seq(self.x_vgg_u),
                                                     self.decoder_kappa_seq(self.x_vgg_u)]))

        self.full_model.compile(optimizer='adam', loss=self._cvae_elbo_loss_tf)

        self.decoder_input = concatenate([self.x_vgg_decoder, self.u_prior])

        self.decoder_model = Model(inputs=[self.x],
                                   outputs=concatenate([
                                                        # self.decoder_mu_seq(self.u_prior),
                                                        # self.decoder_kappa_seq(self.u_prior)]))
                                                        self.decoder_mu_seq(self.decoder_input),
                                                        self.decoder_kappa_seq(self.decoder_input)]))

    def _encoder_mu_log_sigma(self):

        x_vgg_phi = concatenate([self.x_vgg_encoder, self.phi])

        hidden = Dense(512, activation='relu')(Dense(512, activation='relu')(x_vgg_phi))

        mu_encoder = Dense(self.n_u, activation='linear')(hidden)
        log_sigma_encoder = Dense(self.n_u, activation='linear')(hidden)

        return mu_encoder, log_sigma_encoder

    def _prior_mu_log_sigma(self):

        hidden = Dense(512, activation='relu')(self.x_vgg_prior)

        mu_prior = Dense(self.n_u, activation='linear')(hidden)
        log_sigma_prior = Dense(self.n_u, activation='linear')(hidden)

        return mu_prior, log_sigma_prior

    def _sample_u(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=[self.n_u], mean=0., stddev=1.)
        return mu + K.exp(log_sigma / 2) * eps

    def _sample_normal(self, args):
        mu, log_sigma = args
        eps = K.random_normal(shape=[self.n_u], mean=0., stddev=1.)
        return mu*0 + eps

    def _decoder_net_seq(self):
        decoder_mu = Sequential()
        decoder_mu.add(Dense(512, activation='relu',input_shape=[self.x_vgg_shape + self.n_u]))
        # decoder_mu.add(Dense(512, activation='relu', input_shape=[self.n_u]))
        decoder_mu.add(Dense(512, activation='relu'))
        decoder_mu.add(Dense(2, activation='linear'))
        decoder_mu.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

        decoder_kappa = Sequential()
        decoder_kappa.add(Dense(512, activation='relu', input_shape=[self.x_vgg_shape + self.n_u]))
        # decoder_kappa.add(Dense(512, activation='relu', input_shape=[self.n_u]))
        decoder_kappa.add(Dense(512, activation='relu'))
        decoder_kappa.add(Dense(1, activation='linear'))
        decoder_kappa.add(Lambda(lambda x: K.abs(x)))
        return decoder_mu, decoder_kappa

    def _cvae_elbo_loss_tf(self, y_true, model_output):
        mu_prior = model_output[:, 0:self.n_u]
        log_sigma_prior = model_output[:, self.n_u:self.n_u*2]
        mu_encoder = model_output[:, self.n_u*2:self.n_u*3]
        log_sigma_encoder = model_output[:, self.n_u*3:self.n_u*4]
        mu_pred = model_output[:, self.n_u*4:self.n_u*4+2]
        kappa_pred = model_output[:, self.n_u*4+2:]
        reconstruction_err = von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred, input_type='biternion')
        kl = gaussian_kl_divergence_tf(mu_encoder, log_sigma_encoder, mu_prior, log_sigma_prior)
        elbo = self.rec_weight*reconstruction_err - self.kl_weight*kl
        return K.mean(-elbo)

    def _cvae_elbo_loss_np(self, y_true, y_pred):
        mu_prior = y_pred[:, 0:self.n_u]
        log_sigma_prior = y_pred[:, self.n_u:self.n_u*2]
        mu_encoder = y_pred[:, self.n_u*2:self.n_u*3]
        log_sigma_encoder = y_pred[:, self.n_u*3:self.n_u*4]
        mu_pred = y_pred[:, self.n_u*4:self.n_u*4+2]
        kappa_pred = y_pred[:, self.n_u*4+2:]
        log_likelihood = von_mises_log_likelihood_np(y_true, mu_pred, kappa_pred, input_type='biternion')
        kl = gaussian_kl_divergence_np(mu_encoder, log_sigma_encoder, mu_prior, log_sigma_prior)
        elbo = log_likelihood - kl
        return elbo, log_likelihood, kl

    def evaluate(self, x, ytrue_deg, data_part, verbose=1):

        ytrue_bit = deg2bit(ytrue_deg)

        results = dict()

        cvae_preds = self.full_model.predict([x, ytrue_bit])
        elbo, ll, kl = self._cvae_elbo_loss_np(ytrue_bit, cvae_preds)

        results['elbo'] = np.mean(elbo)
        results['elbo_sem'] = sem(elbo)

        results['kl'] = np.mean(kl)
        results['kl_sem'] = sem(kl)

        results['log_likelihood'] = np.mean(ll)
        results['log_likelihood_loss_sem'] = sem(ll)

        # ypreds = self.decoder_model.predict(x)
        # ypreds_bit = ypreds[:, 0:2]
        # kappa_preds = ypreds[:, 2:]

        ypreds_bit = cvae_preds[:, self.n_u*4:self.n_u*4+2]

        ypreds_deg = bit2deg(ypreds_bit)

        loss = maad_from_deg(ytrue_deg, ypreds_deg)
        results['maad_loss'] = np.mean(loss)
        results['maad_loss_sem'] = sem(loss)

        # log_likelihood_loss = von_mises_log_likelihood_np(ytrue_bit, ypreds_bit, kappa_preds,
        #                                                   input_type='biternion')

        # results['log_likelihood'] = np.mean(log_likelihood_loss)
        # results['log_likelihood_loss_sem'] = sem(log_likelihood_loss)

        if verbose:

            print("MAAD error (%s) : %f ± %fSEM" % (data_part, results['maad_loss'], results['maad_loss_sem']))

            print("ELBO (%s) : %f ± %fSEM" % (data_part, results['elbo'], results['elbo_sem']))

            print("KL-div (%s) : %f ± %fSEM" % (data_part, results['kl'], results['kl_sem']))

            # print("log-likelihood (%s) : %f±%fSEM" % (data_part,
            #                                           results['log_likelihood'],
            #                                           results['log_likelihood_loss_sem']))
        return results
