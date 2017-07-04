
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Input, Dense, Lambda, Flatten, Activation, Merge, Concatenate, Add
from keras import layers
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.models import load_model

from models import vgg
from models.cvae import CVAE
from utils.angles import deg2bit, bit2deg
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, maad_from_deg
from utils.losses import gaussian_kl_divergence_tf, gaussian_kl_divergence_np
from utils.losses  import von_mises_log_likelihood_tf, von_mises_log_likelihood_np
from utils.towncentre import load_towncentre
from utils.experiements import get_experiment_id


# #### TownCentre data

# In[2]:

xtr, ytr_deg, xval, yval_deg, xte, yte_deg = load_towncentre('data/TownCentre.pkl.gz', 
                                                             canonical_split=True,
                                                             verbose=1)
image_height, image_width = xtr.shape[1], xtr.shape[2]
ytr_bit = deg2bit(ytr_deg)
yval_bit = deg2bit(yval_deg)
yte_bit = deg2bit(yte_deg)

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

# In[3]:

n_u = 8

cvae_model = CVAE(n_hidden_units=n_u)


# #### Training

# In[5]:

import keras
from utils.custom_keras_callbacks import SideModelCheckpoint

cvae_best_ckpt_path = 'logs/cvae.full_model.best.weights.hdf5'


model_ckpt_callback = keras.callbacks.ModelCheckpoint(cvae_best_ckpt_path,
                                                      monitor='val_loss',
                                                      mode='min',
                                                      save_best_only=True,
                                                      save_weights_only=True,  
                                                      verbose=1)


# In[ ]:

cvae_model.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=10, epochs=50, validation_data=([xval, yval_bit], yval_bit),
                   callbacks=[model_ckpt_callback])


# #### Predictions using decoder part
# 
# $ \phi_i = \mu(x_i,u_i,\theta'') $

# In[ ]:

cvae_best = CVAE(n_hidden_units=n_u)
cvae_best.full_model.load_weights(cvae_best_ckpt_path)


# In[ ]:

from scipy.stats import sem

def evaluate(cvae_model, x, ytrue_deg, data_part):
    
    n_samples = x.shape[0]
    
    ytrue_bit = deg2bit(ytrue_deg)
    
    cvae_preds = cvae_model.full_model.predict([x, ytrue_bit])
    elbo_te, ll_te, kl_te = cvae_model._cvae_elbo_loss_np(ytrue_bit, cvae_preds)

    ypreds = cvae_model.decoder_model.predict(x)
    ypreds_bit = ypreds[:,0:2]
    kappa_preds_te = ypreds[:,2:]

    ypreds_deg = bit2deg(ypreds_bit)

    loss_te = maad_from_deg(ytrue_deg, ypreds_deg)
    mean_loss_te = np.mean(loss_te)
    std_loss_te = np.std(loss_te)

    print("MAAD error (test) : %f ± %f" % (mean_loss_te, std_loss_te))

    # print("kappa (test) : %f ± %f" % (np.mean(kappa_preds_te), np.std(kappa_preds_te)))

    log_likelihood_loss = von_mises_log_likelihood_np(ytrue_bit, ypreds_bit, kappa_preds_te,
                                                         input_type='biternion')

    print("ELBO (%s) : %f ± %f SEM" % (data_part, np.mean(-elbo_te), sem(-elbo_te)))

    # print("KL(encoder|prior) (%s) : %f ± %f SEM" % (data_part, np.mean(-kl_te), sem(-kl_te)))

    print("log-likelihood (%s) : %f±%fSEM" % (data_part, 
                                              np.mean(log_likelihood_loss), 
                                              sem(log_likelihood_loss)))
    return


# In[ ]:

evaluate(cvae_best, xtr, ytr_deg, 'train')


# In[ ]:

evaluate(cvae_best, xval, yval_deg, 'validation')


# In[ ]:

evaluate(cvae_best, xte, yte_deg, 'test')

