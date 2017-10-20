import numpy as np
import keras

from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model, Sequential

import keras.backend as K

from models import vgg

from utils.losses import gaussian_kl_divergence_tf, gaussian_kl_divergence_np
from utils.losses import von_mises_log_likelihood_tf, von_mises_log_likelihood_np
from utils.angles import deg2bit, bit2deg, bit2deg_multi, rad2bit
from utils.losses import maad_from_deg, maximum_expected_utility, importance_loglikelihood
from scipy.stats import sem


class CVAE:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 **kwargs):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.phi_shape = 2
        self.hyp_params = kwargs
        self.n_u = kwargs.get('n_hidden_units', 8)
        self.learning_rate = kwargs.get('learning_rate', 1.0e-3)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1.0e-7)
        self.conv_dropout = kwargs.get('conv_dropout', 0.2)
        self.fc_dropout = kwargs.get('fc_dropout', 0.5)
        self.vgg_fc_layer_size = kwargs.get('vgg_fc_layer_size', 512)
        self.cvae_fc_layer_size = kwargs.get('cvae_fc_layer_size', 512)
        self.kl_weight = kwargs.get('kl_weight', 1.0)

        self.x = Input(shape=[self.image_height, self.image_width, self.n_channels])

        self.phi = Input(shape=[self.phi_shape])

        self.u = Input(shape=[self.n_u])

        self.x_vgg = vgg.vgg_model(image_height=self.image_height,
                                   image_width=self.image_width,
                                   fc_dropout_val=self.fc_dropout,
                                   conv_dropout_val=self.conv_dropout,
                                   fc_layer_size=self.vgg_fc_layer_size)(self.x)

        # self.x_vgg = vgg.vgg_model(n_outputs=self.n_u,
        #                            final_layer=True,
        #                            image_height=self.image_height,
        #                            image_width=self.image_width,
        #                            fc_dropout_val=self.fc_dropout,
        #                            conv_dropout_val=self.conv_dropout,
        #                            fc_layer_size=self.vgg_fc_layer_size)(self.x)

        self.x_vgg_shape = self.x_vgg.get_shape().as_list()[1]

        # self.x_vgg_prior = vgg.vgg_model(image_height=self.image_height,
        #                                  image_width=self.image_width)(self.x)
        #
        # self.x_vgg_decoder = vgg.vgg_model(image_height=self.image_height,
        #                                    image_width=self.image_width)(self.x)

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

        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate,
                                               beta_1=self.beta1,
                                               beta_2=self.beta2,
                                               epsilon=self.epsilon)

        self.full_model.compile(optimizer=self.optimizer, loss=self._cvae_elbo_loss_tf)

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

        hidden = Dense(self.cvae_fc_layer_size, activation='relu')(self.x_vgg)

        mu_prior = Dense(self.n_u, activation='linear')(hidden)
        log_var_prior = Dense(self.n_u, activation='linear')(hidden)

        return mu_prior, log_var_prior

    def _sample_u(self, args):
        mu, log_var = args
        eps = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
        return mu + K.exp(log_var / 2) * eps

    def _decoder_net_seq(self):
        decoder_mu = Sequential()
        decoder_mu.add(Dense(self.cvae_fc_layer_size, activation='relu', input_shape=[self.x_vgg_shape + self.n_u],
                             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)))
        decoder_mu.add(Dense(2, activation='linear',
                             kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)))
        decoder_mu.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

        decoder_kappa = Sequential()
        decoder_kappa.add(Dense(self.cvae_fc_layer_size,
                                activation='relu',
                                input_shape=[self.x_vgg_shape + self.n_u],
                                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)))
        decoder_kappa.add(Dense(1, activation='linear',
                                kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)))
        decoder_kappa.add(Lambda(lambda x: K.abs(x)))
        return decoder_mu, decoder_kappa

    def _cvae_elbo_loss_tf(self, y_true, model_output):
        mu_prior = model_output[:, 0:self.n_u]
        log_var_prior = model_output[:, self.n_u:self.n_u*2]
        mu_encoder = model_output[:, self.n_u*2:self.n_u*3]
        log_var_encoder = model_output[:, self.n_u*3:self.n_u*4]
        mu_pred = model_output[:, self.n_u*5:self.n_u*5+2]
        kappa_pred = model_output[:, self.n_u*5+2:]
        reconstruction_err = von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred)
        kl = gaussian_kl_divergence_tf(mu_encoder, log_var_encoder, mu_prior, log_var_prior)
        elbo = reconstruction_err - self.kl_weight * kl
        return K.mean(-elbo)

    def _cvae_elbo_loss_np(self, y_true, y_pred):
        mu_prior = y_pred[:, 0:self.n_u]
        log_var_prior = y_pred[:, self.n_u:self.n_u*2]
        mu_encoder = y_pred[:, self.n_u*2:self.n_u*3]
        log_var_encoder = y_pred[:, self.n_u*3:self.n_u*4]
        mu_pred = y_pred[:, self.n_u*5:self.n_u*5+2]
        kappa_pred = y_pred[:, self.n_u*5+2:]
        reconstruction_err = von_mises_log_likelihood_np(y_true, mu_pred, kappa_pred)
        kl = gaussian_kl_divergence_np(mu_encoder, log_var_encoder, mu_prior, log_var_prior)
        elbo = reconstruction_err - kl
        return elbo, reconstruction_err, kl

    def get_full_output(self, x, y):
        output = dict()
        y_pred = self.full_model.predict([x, y])
        output['mu_prior'] = y_pred[:, 0:self.n_u]
        output['log_sigma_prior'] = y_pred[:, self.n_u:self.n_u*2]
        output['mu_encoder'] = y_pred[:, self.n_u*2:self.n_u*3]
        output['log_sigma_encoder'] = y_pred[:, self.n_u*3:self.n_u*4]
        output['u_encoder_samples'] = y_pred[:, self.n_u*4:self.n_u*5]
        output['mu_pred'] = y_pred[:, self.n_u*5:self.n_u*5+2]
        output['kappa_pred'] = y_pred[:, self.n_u*5+2:self.n_u*5+3]
        return output

    def fit(self, train_data, val_data, n_epochs, batch_size, callbacks,
            kl_annealing=False):

        xtr, ytr_bit,  = train_data
        xval, yval_bit = val_data

        if kl_annealing:
            kl_weights = np.arange(0.1, 1.0, 0.1)
            for kl_weight in kl_weights:
                self.kl_weight = kl_weight

                self.full_model.compile(optimizer=self.optimizer, loss=self._cvae_elbo_loss_tf)

                print("\n current kl weight: %f \n" % self.kl_weight)
                self.full_model.fit([xtr, ytr_bit], [ytr_bit],
                                    batch_size=batch_size,
                                    epochs=5,
                                    validation_data=([xval, yval_bit], yval_bit),
                                    callbacks=callbacks)

        self.full_model.fit([xtr, ytr_bit], [ytr_bit],
                            batch_size=batch_size,
                            epochs=n_epochs,
                            validation_data=([xval, yval_bit], yval_bit),
                            callbacks=callbacks)

        return

    def save_weights(self, ckpt_path):

        self.full_model.save_weights(ckpt_path)

        return

    def load_weights(self, ckpt_path):

        self.full_model.load_weights(ckpt_path)

        return

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

            n_points = x.shape[0]

            mu_bit_preds = np.zeros([n_points, n_samples, 2])
            kappa_preds  = np.zeros([n_points, n_samples, 1])
            reconstruction_errs = np.zeros([n_points, n_samples, 1])
            kl_preds = np.zeros([n_points, n_samples, 1])
            elbo_preds = np.zeros([n_points, n_samples, 1])
            u_encoder = np.zeros([n_points, n_samples, self.n_u])

            mu_bit_preds_dec = np.zeros([n_points, n_samples, 2])
            kappa_preds_dec = np.zeros([n_points, n_samples, 1])

            for sid in range(0, n_samples):
                preds = self.full_model.predict([x, y_bit])
                mu_prior = preds[:, 0:self.n_u]
                log_sigma_prior = preds[:, self.n_u:self.n_u*2]
                mu_encoder = preds[:, self.n_u*2:self.n_u*3]
                log_sigma_encoder = preds[:, self.n_u*3:self.n_u*4]
                u_encoder[:, sid, :] = preds[:, self.n_u*4:self.n_u*5]
                mu_bit_preds[:, sid, :] = preds[:, self.n_u * 5:self.n_u * 5 + 2]
                kappa_preds[:, sid, :] = preds[:, self.n_u * 5 + 2:].reshape(-1, 1)
                elbo, reconstruction, kl = self._cvae_elbo_loss_np(y_bit, preds)
                reconstruction_errs[:, sid, :] = reconstruction
                kl_preds[:, sid, :] = kl
                elbo_preds[:, sid, :] = elbo
                preds_dec = self.decoder_model.predict(x, batch_size=100)
                mu_bit_preds_dec[:, sid, :] = preds_dec[:, 0:2]
                kappa_preds_dec[:, sid, :] = preds_dec[:, 2:].reshape(-1, 1)

            preds = dict()

            preds['mu_encoder'] = mu_encoder
            preds['log_sigma_encoder'] = log_sigma_encoder
            preds['mu_prior'] = mu_prior
            preds['log_sigma_prior'] = log_sigma_prior
            preds['u_encoder'] = u_encoder
            preds['mu_bit'] = mu_bit_preds
            preds['kappa'] = kappa_preds
            preds['reconstruction_err'] = reconstruction_errs
            preds['kl_div'] = kl_preds
            preds['elbo'] = elbo_preds
            preds['mu_bit_dec'] = mu_bit_preds_dec
            preds['kappa_dec'] = kappa_preds_dec
            preds['mu_rad_dec'] = np.deg2rad(bit2deg_multi(preds['mu_bit_dec']))
            preds['maxutil_deg_dec'] = maximum_expected_utility(np.rad2deg(preds['mu_rad_dec']))

            return preds

    def get_multiple_decoder_predictions(self, x, n_samples=5):

            n_points = x.shape[0]

            mu_bit_preds_dec = np.zeros([n_points, n_samples, 2])
            kappa_preds_dec = np.zeros([n_points, n_samples, 1])

            for sid in range(0, n_samples):
                preds_dec = self.decoder_model.predict(x, batch_size=100)
                mu_bit_preds_dec[:, sid, :] = preds_dec[:, 0:2]
                kappa_preds_dec[:, sid, :] = preds_dec[:, 2:].reshape(-1, 1)

            preds = dict()
            preds['mu_bit'] = mu_bit_preds_dec
            preds['kappa'] = kappa_preds_dec
            preds['mu_rad'] = np.deg2rad(bit2deg_multi(preds['mu_bit']))
            preds['maxutil_deg'] = maximum_expected_utility(np.rad2deg(preds['mu_rad']))

            return preds

    def evaluate(self, x, ytrue_deg, data_part, n_samples=50, verbose=1,
                 return_per_image=False):

        ytrue_bit = deg2bit(ytrue_deg)

        results = dict()

        preds = self.get_multiple_predictions(x, ytrue_bit, n_samples=n_samples)

        results['elbo'] = np.mean(preds['elbo'])
        results['elbo_sem'] = sem(np.mean(preds['elbo'], axis=1))

        results['kl_div'] = np.mean(preds['kl_div'])
        results['kl_div_sem'] = sem(np.mean(preds['kl_div'], axis=1))

        ypreds = self.decoder_model.predict(x)
        ypreds_bit = ypreds[:, 0:2]
        kappa_preds = ypreds[:, 2:]

        ypreds_deg = bit2deg(ypreds_bit)

        loss = maad_from_deg(ytrue_deg, preds['maxutil_deg_dec'])
        results['maad_loss'] = np.mean(loss)
        results['maad_loss_sem'] = sem(loss, axis=None)

        importance_loglikelihoods = importance_loglikelihood(preds['mu_encoder'], preds['log_sigma_encoder'],
                                                     preds['mu_prior'], preds['log_sigma_prior'],
                                                     preds['u_encoder'],
                                                     preds['mu_bit'], preds['kappa'],
                                                     ytrue_bit)

        results['importance_log_likelihood'] = np.mean(importance_loglikelihoods)
        results['importance_log_likelihood_sem'] = sem(importance_loglikelihoods, axis=None)

        if verbose:

            print("MAAD error (%s) : %f pm %fSEM" % (data_part, results['maad_loss'], results['maad_loss_sem']))

            print("ELBO (%s) : %f pm %fSEM" % (data_part, results['elbo'], results['elbo_sem']))

            print("Approx Log-Likelihood, importance sampling (%s) : %f pm %fSEM" %
                  (data_part, results['importance_log_likelihood'], results['importance_log_likelihood_sem']))

            print("KL-div (%s) : %f pm%fSEM" % (data_part, results['kl_div'], results['kl_div_sem']))

        if return_per_image:
            results['point_preds'] = bit2deg(deg2bit(preds['maxutil_deg_dec']))
            results['maad'] = loss
            results['importance_log_likelihood'] = importance_loglikelihoods

        return results

    def pdf(self, x, x_vals, n_samples=10):
        """ Compute probability density function on a circle given images

        Parameters
        ----------
        x: numpy array of shape [n_images, image_width, image_height, n_channels]
            angles in biternion (cos, sin) representation that will be used to compute likelihood

        x_vals: numpy array of shape [n_points]
            angles (in rads) at which pdf values were computed

        Returns
        -------

        pdfs: numpy array of shape [n_images, n_samples, n_points]
            array containing pdf values for each CVAE sample on circle [0, 2pi] for each values

        acc_pdf: numpy array of shape [n_images, n_points]
            array containing accumulated pdf value on circle [0, 2pi] for each values
        """

        n_images = x.shape[0]

        x_vals_tiled = np.ones(n_images)

        decoder_preds = self.get_multiple_decoder_predictions(x, n_samples=n_samples)

        vm_pdfs = np.zeros([n_images, n_samples, len(x_vals)])

        for xid, xval in enumerate(x_vals):

            for sid in range(0, n_samples):

                x_bit = rad2bit(x_vals_tiled*xval)

                vm_pdfs[:, sid, xid] = np.exp(np.squeeze(von_mises_log_likelihood_np(x_bit,
                                                                          decoder_preds['mu_bit'][:, sid, :],
                                                                          decoder_preds['kappa'][:, sid])))

        acc_pdf = np.mean(vm_pdfs, axis=1)
        return vm_pdfs, acc_pdf

    def pdf_importance(self, x, y_bit, x_vals, n_samples=10):
        """ Compute probability density function on a circle given images

        Parameters
        ----------
        x: numpy array of shape [n_images, image_width, image_height, n_channels]
            angles in biternion (cos, sin) representation that will be used to compute likelihood

        x_vals: numpy array of shape [n_points]
            angles (in rads) at which pdf values were computed

        Returns
        -------

        pdfs: numpy array of shape [n_images, n_samples, n_points]
            array containing pdf values for each CVAE sample on circle [0, 2pi] for each values

        acc_pdf: numpy array of shape [n_images, n_points]
            array containing accumulated pdf value on circle [0, 2pi] for each values
        """

        n_images = x.shape[0]

        x_vals_tiled = np.ones(n_images)

        decoder_preds = self.get_multiple_decoder_predictions(x, n_samples=n_samples)

        vm_pdfs = np.zeros([n_images, n_samples, len(x_vals)])

        for xid, xval in enumerate(x_vals):

            for sid in range(0, n_samples):

                x_bit = rad2bit(x_vals_tiled*xval)

                vm_pdfs[:, sid, xid] = np.exp(np.squeeze(von_mises_log_likelihood_np(x_bit,
                                                                          decoder_preds['mu_bit'][:, sid, :],
                                                                          decoder_preds['kappa'][:, sid])))

        acc_pdf = np.mean(vm_pdfs, axis=1)
        return vm_pdfs, acc_pdf

