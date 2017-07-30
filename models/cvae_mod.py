import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model, Sequential

import keras.backend as K

from models import vgg

from utils.losses import gaussian_kl_divergence_tf, gaussian_kl_divergence_np
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np
from utils.losses import gaussian_log_likelihood_np, gaussian_log_likelihood_tf
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
                 n_samples=1):

        self.n_u = n_hidden_units
        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.phi_shape = 2
        self.kl_weight = kl_weight
        self.n_samples = n_samples

        self.x = Input(shape=[self.image_height, self.image_width, self.n_channels])
        self.phi = Input(shape=[self.phi_shape])
        self.u = Input(shape=[self.n_u])

        self.x_vgg = vgg.vgg_model(image_height=self.image_height,
                                   image_width=self.image_width)(self.x)

        self.x_vgg_shape = self.x_vgg.get_shape().as_list()[1]

        self.mu_encoder, self.log_sigma_encoder = self._encoder_mu_log_sigma()

        self.mu_prior, self.log_sigma_prior = self._prior_mu_log_sigma()

        self.u_prior = Lambda(self._sample_u)([self.mu_prior, self.log_sigma_prior])

        self.decoder_mu_seq, self.decoder_kappa_seq = self._decoder_net_seq()

        self.u_encoder_list = []
        self.x_vgg_u_list = []
        self.dec_mu_list = []
        self.dec_kappa_list = []

        for i in range(0, self.n_samples):
            u_enc = Lambda(self._sample_u)([self.mu_encoder, self.log_sigma_encoder])
            self.u_encoder_list.append(u_enc)
            x_u = concatenate([self.x_vgg, u_enc])
            self.x_vgg_u_list.append(x_u)
            dec_mu_out = self.decoder_mu_seq(x_u)
            self.dec_mu_list.append(dec_mu_out)
            dec_kappa_out = self.decoder_kappa_seq(x_u)
            self.dec_kappa_list.append(dec_kappa_out)

        if self.n_samples > 1:
            self.u_samples = concatenate(self.u_encoder_list)
            self.dec_mus = concatenate(self.dec_mu_list)
            self.dec_kappas = concatenate(self.dec_kappa_list)
        else:
            self.u_samples = self.u_encoder_list[0]
            self.dec_mus = self.dec_mu_list[0]
            self.dec_kappas = self.dec_kappa_list[0]

        self.full_model = Model(inputs=[self.x, self.phi],
                                outputs=concatenate([self.mu_prior,
                                                     self.log_sigma_prior,
                                                     self.mu_encoder,
                                                     self.log_sigma_encoder,
                                                     self.u_samples,
                                                     self.dec_mus,
                                                     self.dec_kappas]))

        self.full_model.compile(optimizer='adam', loss=self.importance_log_likelihood_tf)

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
        log_sigma_encoder = Dense(self.n_u, activation='linear')(hidden)

        return mu_encoder, log_sigma_encoder

    def _prior_mu_log_sigma(self):

        hidden = Dense(512, activation='relu')(self.x_vgg)

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
        mu_pred = model_output[:, self.n_u*5:self.n_u*5+2]
        kappa_pred = model_output[:, self.n_u*5+2:]
        log_likelihood = von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred)
        kl = gaussian_kl_divergence_tf(mu_encoder, log_sigma_encoder, mu_prior, log_sigma_prior)
        elbo = log_likelihood - self.kl_weight*kl
        return K.mean(-elbo)

    def _cvae_elbo_loss_np(self, y_true, y_pred):
        mu_prior = y_pred[:, 0:self.n_u]
        log_sigma_prior = y_pred[:, self.n_u:self.n_u*2]
        mu_encoder = y_pred[:, self.n_u*2:self.n_u*3]
        log_sigma_encoder = y_pred[:, self.n_u*3:self.n_u*4]
        mu_pred = y_pred[:, self.n_u*5:self.n_u*5+2]
        kappa_pred = y_pred[:, self.n_u*5+2:]
        log_likelihood = von_mises_log_likelihood_np(y_true, mu_pred, kappa_pred)
        kl = gaussian_kl_divergence_np(mu_encoder, log_sigma_encoder, mu_prior, log_sigma_prior)
        elbo = log_likelihood - kl
        return elbo, log_likelihood, kl

    def parse_output_np(self, y_pred):

        output = dict()

        output['mu_prior'] = y_pred[:, 0:self.n_u]
        output['log_sigma_prior'] = y_pred[:, self.n_u:self.n_u*2]
        output['mu_encoder'] = y_pred[:, self.n_u*2:self.n_u*3]
        output['log_sigma_encoder'] = y_pred[:, self.n_u*3:self.n_u*4]
        u_samples_flat = y_pred[:, self.n_u*4:self.n_u*(4+self.n_samples)]
        output['u_samples'] = np.reshape(u_samples_flat, [-1, self.n_samples, self.n_u])

        muvm_ptr = self.n_u*(4+self.n_samples)
        kappavm_ptr = self.n_u*(4+self.n_samples)+2*self.n_samples

        mu_preds_flat = y_pred[:, muvm_ptr:muvm_ptr+2*self.n_samples]
        output['mu_preds'] = np.reshape(mu_preds_flat, [-1, self.n_samples, 2])
        kappa_preds_flat = y_pred[:, kappavm_ptr:kappavm_ptr+self.n_samples]
        output['kappa_preds'] = np.reshape(kappa_preds_flat, [-1, self.n_samples, 1])

        return output

    def parse_output_tf(self, y_pred):

        output = dict()

        output['mu_prior'] = y_pred[:, 0:self.n_u]
        output['log_sigma_prior'] = y_pred[:, self.n_u:self.n_u*2]
        output['mu_encoder'] = y_pred[:, self.n_u*2:self.n_u*3]
        output['log_sigma_encoder'] = y_pred[:, self.n_u*3:self.n_u*4]
        u_samples_flat = y_pred[:, self.n_u*4:self.n_u*(4+self.n_samples)]
        output['u_samples'] = K.reshape(u_samples_flat, [-1, self.n_samples, self.n_u])

        muvm_ptr = self.n_u*(4+self.n_samples)
        kappavm_ptr = self.n_u*(4+self.n_samples)+2*self.n_samples

        mu_preds_flat = y_pred[:, muvm_ptr:muvm_ptr+2*self.n_samples]
        output['mu_preds'] = K.reshape(mu_preds_flat, [-1, self.n_samples, 2])
        kappa_preds_flat = y_pred[:, kappavm_ptr:kappavm_ptr+self.n_samples]
        output['kappa_preds'] = K.reshape(kappa_preds_flat, [-1, self.n_samples, 1])

        return output

    def importance_log_likelihood_np(self, y_true, y_preds):
        """ Compute importance log-likelihood for CVAE-based Von-Mises mixture model

        Parameters
        ----------

        y_true: numpy array of size [n_points, 2]
            samples of angle phi (cos, sin) that would be used to compute Von-Mises log-likelihood
        y_preds: numpy array of size [n_points, n_outputs]
            full output of the CVAE model (prior, encoder, decoder)

        Returns
        -------

        importance_log_likelihood: numpy array of size [n_points]
            log-likelihood estimation for points based on samples
        """

        out_parsed = self.parse_output_np(y_preds)

        u_samples = out_parsed['u_samples']
        mu_encoder = out_parsed['mu_encoder']
        std_encoder = np.exp(out_parsed['log_sigma_encoder']/2)
        mu_prior = out_parsed['mu_prior']
        std_prior = np.exp(out_parsed['log_sigma_prior']/2)
        mu_decoder = out_parsed['mu_preds']
        kappa_decoder = out_parsed['kappa_preds']

        n_points, n_samples, _ = u_samples.shape

        vm_likelihood = np.zeros([n_points, n_samples])

        for sid in range(0, n_samples):
            vm_likelihood[:, sid] = np.squeeze(np.exp(von_mises_log_likelihood_np(y_true, mu_decoder[:,sid,:], kappa_decoder[:,sid,:])))

        prior_log_likelihood = gaussian_log_likelihood_np(mu_prior, std_prior, u_samples)
        encoder_log_likelihood = gaussian_log_likelihood_np(mu_encoder, std_encoder, u_samples)

        sample_weight = np.exp(prior_log_likelihood - encoder_log_likelihood)

        importance_log_likelihood = np.log(np.mean(vm_likelihood*sample_weight, axis=1))

        return importance_log_likelihood

    def importance_log_likelihood_tf(self, y_true, y_preds):
        """ Compute importance log-likelihood for CVAE-based Von-Mises mixture model

        Parameters
        ----------

        y_true: Tensor of size [n_points, 2]
            true values of angle phi (cos, sin) that would be used to compute Von-Mises log-likelihood
        y_preds: Tensor of size [n_points, n_outputs]
            full output of the CVAE model (prior, encoder, decoder)

        Returns
        -------

        importance_log_likelihood: numpy array of size [n_points]
            log-likelihood estimation for points based on samples
        """

        out_parsed = self.parse_output_tf(y_preds)

        u_samples = out_parsed['u_samples']
        mu_encoder = out_parsed['mu_encoder']
        std_encoder = tf.exp(out_parsed['log_sigma_encoder']/2)
        mu_prior = out_parsed['mu_prior']
        std_prior = tf.exp(out_parsed['log_sigma_prior']/2)
        mu_decoder = out_parsed['mu_preds']
        kappa_decoder = out_parsed['kappa_preds']

        n_points, n_samples, _ = u_samples.shape

        vm_likelihoods = []

        for sid in range(0, n_samples):
            vm_likelihood = tf.exp(von_mises_log_likelihood_tf(y_true, mu_decoder[:,sid,:], kappa_decoder[:,sid,:]))
            vm_likelihoods.append(vm_likelihood)

        vm_likelihoods = tf.squeeze(tf.stack(vm_likelihoods, axis=1), axis=2)

        prior_log_likelihood = gaussian_log_likelihood_tf(mu_prior, std_prior, u_samples)
        encoder_log_likelihood = gaussian_log_likelihood_tf(mu_encoder, std_encoder, u_samples)

        sample_weight = tf.exp(prior_log_likelihood - encoder_log_likelihood)

        importance_log_likelihood = tf.log(tf.reduce_mean(vm_likelihoods*sample_weight, axis=1))

        # log_likelihood = von_mises_log_likelihood_tf(y_true, mu_decoder[:, 0, :], kappa_decoder[:, 0, :])
        kl = gaussian_kl_divergence_tf(mu_encoder, out_parsed['log_sigma_encoder'],
                                       mu_prior, out_parsed['log_sigma_prior'])

        # elbo = log_likelihood - kl

        return K.mean(-importance_log_likelihood + kl)

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
