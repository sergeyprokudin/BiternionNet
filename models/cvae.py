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
                 kl_weight=1.0):

        self.n_u = n_hidden_units
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.phi_shape = 2
        self.kl_weight = kl_weight

        self.x = Input(shape=[self.image_height, self.image_width, self.n_channels])
        self.phi = Input(shape=[self.phi_shape])
        self.u = Input(shape=[self.n_u])

        self.x_vgg = vgg.vgg_model(image_height=self.image_height,
                                   image_width=self.image_width)(self.x)

        # self.x_vgg_prior = vgg.vgg_model(image_height=self.image_height,
        #                                  image_width=self.image_width)(self.x)
        #
        # self.x_vgg_decoder = vgg.vgg_model(image_height=self.image_height,
        #                                    image_width=self.image_width)(self.x)

        self.x_vgg_shape = self.x_vgg.get_shape().as_list()[1]

        self.mu_encoder, self.log_var_encoder = self._encoder_mu_log_sigma()

        self.mu_prior, self.log_var_prior = self._prior_mu_log_sigma()

        self.u_prior = Lambda(self._sample_u)([self.mu_prior, self.log_var_prior])
        # self.u_prior = Lambda(self._sample_normal)([self.mu_prior, self.log_sigma_prior])
        self.u_encoder = Lambda(self._sample_u)([self.mu_encoder, self.log_var_encoder])

        self.x_vgg_enc_u = concatenate([self.x_vgg, self.u_encoder])

        self.decoder_mu_seq, self.decoder_kappa_seq = self._decoder_net_seq()

        self.full_model = Model(inputs=[self.x, self.phi],
                                outputs=concatenate([self.mu_prior,
                                                     self.log_var_prior,
                                                     self.mu_encoder,
                                                     self.log_var_encoder,
                                                     self.u_encoder,
                                                     # self.decoder_mu_seq(self.u_encoder),
                                                     # self.decoder_kappa_seq(self.u_encoder)]))
                                                     self.decoder_mu_seq(self.x_vgg_enc_u),
                                                     self.decoder_kappa_seq(self.x_vgg_enc_u)]))

        self.full_model.compile(optimizer='adam', loss=self._cvae_elbo_loss_tf)

        self.decoder_input = concatenate([self.x_vgg, self.u_prior])

        self.decoder_model = Model(inputs=[self.x],
                                   outputs=concatenate([
                                                        # self.decoder_mu_seq(self.u_prior),
                                                        # self.decoder_kappa_seq(self.u_prior)]))
                                                        self.decoder_mu_seq(self.decoder_input),
                                                        self.decoder_kappa_seq(self.decoder_input)]))

    def _encoder_mu_log_sigma(self):

        x_vgg_phi = concatenate([self.x_vgg, self.phi])

        hidden = Dense(512, activation='relu')(Dense(512, activation='relu')(x_vgg_phi))

        mu_encoder = Dense(self.n_u, activation='linear')(hidden)
        log_var_encoder = Dense(self.n_u, activation='linear')(hidden)

        return mu_encoder, log_var_encoder

    def _prior_mu_log_sigma(self):

        hidden = Dense(512, activation='relu')(self.x_vgg)

        mu_prior = Dense(self.n_u, activation='linear')(hidden)
        log_var_prior = Dense(self.n_u, activation='linear')(hidden)

        return mu_prior, log_var_prior

    def _sample_u(self, args):
        mu, log_var = args
        eps = K.random_normal(shape=[self.n_u], mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * eps

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
        log_var_prior = model_output[:, self.n_u:self.n_u*2]
        mu_encoder = model_output[:, self.n_u*2:self.n_u*3]
        log_var_encoder = model_output[:, self.n_u*3:self.n_u*4]
        mu_pred = model_output[:, self.n_u*5:self.n_u*5+2]
        kappa_pred = model_output[:, self.n_u*5+2:]
        log_likelihood = von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred)
        kl = gaussian_kl_divergence_tf(mu_encoder, log_var_encoder, mu_prior, log_var_prior)
        elbo = log_likelihood - self.kl_weight*kl
        return K.mean(-elbo)

    def _cvae_elbo_loss_np(self, y_true, y_pred):
        mu_prior = y_pred[:, 0:self.n_u]
        log_var_prior = y_pred[:, self.n_u:self.n_u*2]
        mu_encoder = y_pred[:, self.n_u*2:self.n_u*3]
        log_var_encoder = y_pred[:, self.n_u*3:self.n_u*4]
        mu_pred = y_pred[:, self.n_u*5:self.n_u*5+2]
        kappa_pred = y_pred[:, self.n_u*5+2:]
        log_likelihood = von_mises_log_likelihood_np(y_true, mu_pred, kappa_pred)
        kl = gaussian_kl_divergence_np(mu_encoder, log_var_encoder, mu_prior, log_var_prior)
        elbo = log_likelihood - kl
        return elbo, log_likelihood, kl

    def get_full_output(self, x, y):
        output = dict()
        y_pred = self.full_model.predict([x, y])
        output['mu_prior'] = y_pred[:, 0:self.n_u]
        output['log_sigma_prior'] = y_pred[:, self.n_u:self.n_u*2]
        output['mu_encoder'] = y_pred[:, self.n_u*2:self.n_u*3]
        output['log_sigma_encoder'] = y_pred[:, self.n_u*3:self.n_u*4]
        output['u_encoder'] = y_pred[:, self.n_u*4:self.n_u*5]
        output['mu_pred'] = y_pred[:, self.n_u*5:self.n_u*5+2]
        output['kappa_pred'] = y_pred[:, self.n_u*5+2:]
        return output

    def generate_multiple_samples(self, x, n_samples=10):

        n_points = x.shape[0]
        cvae_kappa_preds = np.zeros([n_points, n_samples, 1])
        cvae_mu_preds = np.zeros([n_points, n_samples, 2])

        for i in range(0, n_samples):
            cvae_preds = self.decoder_model.predict(x)
            cvae_mu_preds[:, i, :] = cvae_preds[:, 0:2]
            cvae_kappa_preds[:, i, :] = cvae_preds[:, 2].reshape(-1, 1)

        return cvae_mu_preds, cvae_kappa_preds

    def get_multiple_predictions(self, x, y_bit, n_samples=5):

        mu_rad_preds = np.zeros([n_samples, x.shape[0], 1])
        mu_bit_preds = np.zeros([n_samples, x.shape[0], 2])
        kappa_preds  = np.zeros([n_samples, x.shape[0], 1])
        reconstruction_errs = np.zeros([n_samples, x.shape[0], 1])
        kl_preds = np.zeros([n_samples, x.shape[0], 1])
        elbo_preds = np.zeros([n_samples, x.shape[0], 1])

        mu_rad_preds_dec = np.zeros([n_samples, x.shape[0], 1])
        mu_bit_preds_dec = np.zeros([n_samples, x.shape[0], 2])
        kappa_preds_dec = np.zeros([n_samples, x.shape[0], 1])

        for i in range(0, n_samples):
            preds = self.full_model.predict([x, y_bit], batch_size=500)
            mu_bit_preds[i, :, :] =  preds[:, self.n_u*5:self.n_u*5+2]
            mu_rad_preds[i, :, :] = np.deg2rad(bit2deg(preds[:, self.n_u*4:self.n_u*4+2])).reshape(-1,1)
            kappa_preds[i, :, :] = preds[:, self.n_u*5+2:].reshape(-1,1)
            elbo, reconstruction, kl = self._cvae_elbo_loss_np(y_bit, preds)
            reconstruction_errs[i, :, :] = reconstruction
            kl_preds[i, :, :] = kl
            elbo_preds[i, :, :] = elbo
            preds_dec = self.decoder_model.predict(x, batch_size=500)
            mu_bit_preds_dec[i, :, :] = preds_dec[:, 0:2]
            mu_rad_preds_dec[i, :, :] = np.deg2rad(bit2deg(preds_dec[:, 0:2])).reshape(-1,1)
            kappa_preds_dec[i, :, :] = preds_dec[:, 2:].reshape(-1,1)

        res = dict()

        res['mu_rad_preds'] = mu_rad_preds
        res['mu_bit_preds'] = mu_bit_preds
        res['kappa_preds'] = kappa_preds
        res['recontruction_errs'] = reconstruction_errs
        res['kl_preds'] = kl_preds
        res['elbo_preds'] = elbo_preds
        res['mu_rad_preds_dec'] = mu_rad_preds_dec
        res['mu_bit_preds_dec'] = mu_bit_preds_dec
        res['kappa_preds_dec'] = kappa_preds_dec

        return res

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

        ypreds = self.decoder_model.predict(x)
        ypreds_bit = ypreds[:, 0:2]
        kappa_preds = ypreds[:, 2:]

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
