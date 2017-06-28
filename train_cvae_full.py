
# coding: utf-8

# In[1]:

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
from utils.angles import deg2bit, bit2deg
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.losses import gaussian_kl_divergence_tf, gaussian_kl_divergence_np
from utils.losses  import von_mises_log_likelihood_tf, von_mises_log_likelihood_np
from utils.towncentre import load_towncentre
from utils.experiements import get_experiment_id


# #### TownCentre data

# In[2]:

xtr, ytr_deg, xte, yte_deg = load_towncentre('data/TownCentre.pkl.gz', canonical_split=True)
image_height, image_width = xtr.shape[1], xtr.shape[2]
ytr_bit = deg2bit(ytr_deg)
yte_bit = deg2bit(yte_deg)


# In[3]:

# fig, axs = plt.subplots(1, 10, figsize=(30, 15))
# for i in range(0, 10):
#     axs[i].imshow(xtr[i])


# In[4]:

image_height, image_width, n_channels = xtr.shape[1:]
flatten_x_shape = xtr[0].flatten().shape[0]
phi_shape = yte_bit.shape[1]


# #### Notation
# 
# $x$ - image,
# 
# $\phi$ - head angle,
# 
# $u$ - hidden variable

# #### Prior network
# 
# $ p(u|x) \sim \mathcal{N}(\mu_1(x, \theta), \sigma_1(x, \theta)) $
# 
# #### Encoder network
# 
# $ q(u|x,\phi) \sim \mathcal{N}(\mu_2(x, \theta), \sigma_2(x, \theta)) $
# 
# #### Sample  $u \sim \{p(u|x), q(u|x,\phi) \}$
# 
# #### Decoder network
# 
# $p(\phi|u,x) \sim \mathcal{VM}(\mu(x,u,\theta''), \kappa(x,u,\theta'')) $

# In[5]:

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
        
        self.mu_prior, self.log_sigma_prior = self._prior_mu_log_sigma()
        
        self.u_prior = Lambda(self._sample_u)([self.mu_prior, self.log_sigma_prior])
        self.u_encoder = Lambda(self._sample_u)([self.mu_encoder, self.log_sigma_encoder])

        self.x_vgg_u = concatenate([self.x_vgg, self.u_encoder])

        self.decoder_mu_seq, self.decoder_kappa_seq = self._decoder_net_seq()

        self.full_model = Model(inputs=[self.x, self.phi],
                                outputs=concatenate([self.mu_prior,
                                                     self.log_sigma_prior,
                                                     self.mu_encoder,
                                                     self.log_sigma_encoder,
                                                     self.decoder_mu_seq(self.x_vgg_u),
                                                     self.decoder_kappa_seq(self.x_vgg_u)]))
        
        self.decoder_input = concatenate([self.x_vgg, self.u_prior])
        self.decoder_model = Model(inputs=[self.x],
                                   outputs=concatenate([self.decoder_mu_seq(self.decoder_input),
                                                        self.decoder_kappa_seq(self.decoder_input)]))
        
    def _encoder_mu_log_sigma(self):

        x_vgg_phi = concatenate([self.x_vgg, self.phi])

        hidden = Dense(512, activation='relu')(x_vgg_phi)

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
        decoder_kappa.add(Lambda(lambda x: K.abs(x)))
        
        return decoder_mu, decoder_kappa


# In[6]:

n_u = 8

cvae = CVAE(n_hidden_units=n_u)


# In[7]:

def cvae_loss(y_true, model_output):
    mu_prior = model_output[:, 0:n_u]
    log_sigma_prior = model_output[:, n_u:n_u*2]
    mu_encoder = model_output[:, n_u*2:n_u*3]
    log_sigma_encoder = model_output[:, n_u*3:n_u*4]
    mu_pred = model_output[:, n_u*4:n_u*4+2]
    kappa_pred = model_output[:, n_u*4+2:]
    log_likelihood = von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred, input_type='biternion')
    kl = gaussian_kl_divergence_tf(mu_encoder, log_sigma_encoder, mu_prior, log_sigma_prior)
    return K.mean(-log_likelihood + kl)


# In[8]:

def cvae_elbo_np(y_true, y_pred):
    mu_prior = y_pred[:, 0:n_u]
    log_sigma_prior = y_pred[:, n_u:n_u*2]
    mu_encoder = y_pred[:, n_u*2:n_u*3]
    log_sigma_encoder = y_pred[:, n_u*3:n_u*4]
    mu_pred = y_pred[:, n_u*4:n_u*4+2]
    kappa_pred = y_pred[:, n_u*4+2:]
    log_likelihood = von_mises_log_likelihood_np(y_true, mu_pred, kappa_pred, input_type='biternion')
    kl = gaussian_kl_divergence_np(mu_encoder, log_sigma_encoder, mu_prior, log_sigma_prior)
    loss = -log_likelihood + kl
    return loss, log_likelihood, kl


# #### Training

# In[9]:

cvae = CVAE()
#optimizer = keras.optimizers.Adadelta(lr=0.001)
cvae.full_model.compile(optimizer='adam', loss=cvae_loss)


# In[ ]:

import keras
model_ckpt_callback = keras.callbacks.ModelCheckpoint('cvae_full.h5',
                                                      monitor='val_loss',
                                                      mode='min',
                                                      save_best_only=True,
                                                      verbose=1)


# In[ ]:

cvae.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=10, epochs=20, validation_split=0.1,
                   callbacks=[model_ckpt_callback])


# #### Predictions using decoder part
# 
# $ \phi_i = \mu(x_i,u_i,\theta'') $

# In[ ]:

from scipy.stats import sem

n_samples = xte.shape[0]
#ute = np.random.normal(0,1, [n_samples,n_u])

#yte_cvae_preds = cvae.full_model.predict([xte, yte_bit])

cvae_preds = cvae.full_model.predict([xte, yte_bit])
elbo_te, ll_te, kl_te = cvae_elbo_np(yte_bit, cvae_preds)

yte_preds = cvae.decoder_model.predict(xte)
yte_preds_bit = yte_preds[:,0:2]
kappa_preds_te = yte_preds[:,2:]

yte_preds_deg = bit2deg(yte_preds_bit)

loss_te = maad_from_deg(yte_preds_deg, yte_deg)
mean_loss_te = np.mean(loss_te)
std_loss_te = np.std(loss_te)

print("MAAD error (test) : %f ± %f" % (mean_loss_te, std_loss_te))

#kappa_preds_te = np.ones([xte.shape[0], 1]) 

print("kappa (test) : %f ± %f" % (np.mean(kappa_preds_te), np.std(kappa_preds_te)))

log_likelihood_loss_te = von_mises_log_likelihood_np(yte_bit, yte_preds_bit, kappa_preds_te,
                                                     input_type='biternion')


print("ELBO (test) : %f ± %f SEM" % (np.mean(-elbo_te), sem(-elbo_te)))
# print("log-likelihood (test) : %f ± %f SEM" % (np.mean(-ll_te), sem(-ll_te)))
print("KL(encoder|prior) (test) : %f ± %f SEM" % (np.mean(-kl_te), sem(-kl_te)))

print("log-likelihood (test) : %f±%fSEM" % (np.mean(log_likelihood_loss_te), sem(log_likelihood_loss_te)))


# In[ ]:

n_samples = xtr.shape[0]
#utr = np.random.normal(0,1, [n_samples,n_u])

#ytr_cvae_preds = cvae.full_model.predict([xtr, ytr_bit])

cvae_preds = cvae.full_model.predict([xtr, ytr_bit])
elbo_tr, ll_tr, kl_tr = cvae_elbo_np(ytr_bit, cvae_preds)

ytr_preds = cvae.decoder_model.predict(xtr)
ytr_preds_bit = ytr_preds[:,0:2]
kappa_preds_tr = ytr_preds[:,2:]

ytr_preds_deg = bit2deg(ytr_preds_bit)

loss_tr = maad_from_deg(ytr_preds_deg, ytr_deg)
mean_loss_tr = np.mean(loss_tr)
std_loss_tr = np.std(loss_tr)

print("MAAD error (train) : %f ± %f" % (mean_loss_tr, std_loss_tr))

#kappa_preds_tr = np.ones([xtr.shape[0], 1]) 

print("kappa (train) : %f ± %f" % (np.mean(kappa_preds_tr), np.std(kappa_preds_tr)))

log_likelihood_loss_tr = von_mises_log_likelihood_np(ytr_bit, ytr_preds_bit, kappa_preds_tr,
                                                     input_type='biternion')



print("ELBO (train) : %f ± %f SEM" % (np.mean(-elbo_tr), sem(-elbo_tr)))
# print("log-likelihood (train) : %f ± %f SEM" % (np.mean(-ll_tr), sem(-ll_tr)))
print("KL(encoder|prior) (train) : %f ± %f SEM" % (np.mean(-kl_tr), sem(-kl_tr)))

print("log-likelihood (train) : %f±%fSEM" % (np.mean(log_likelihood_loss_tr), sem(log_likelihood_loss_tr)))

