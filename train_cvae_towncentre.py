
# coding: utf-8

# In[2]:

import keras

from models.cvae import CVAE
from utils.angles import deg2bit, bit2deg
from utils.towncentre import load_towncentre
from utils.custom_keras_callbacks import ModelCheckpointEveryNBatch

# #### TownCentre data

# In[3]:

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

# In[4]:

n_u = 8

cvae_model = CVAE(n_hidden_units=n_u)


# #### Training

# In[5]:


from utils.custom_keras_callbacks import SideModelCheckpoint

cvae_best_ckpt_path = 'logs/cvae.full_model.best.weights.hdf5'


# model_ckpt_callback = keras.callbacks.ModelCheckpoint(cvae_best_ckpt_path,
#                                                       monitor='val_loss',
#                                                       mode='min',
#                                                       save_best_only=True,
#                                                       save_weights_only=True,
#                                                       verbose=1)

model_ckpt_callback = ModelCheckpointEveryNBatch(cvae_best_ckpt_path,
                                                 xval=[xval, yval_bit],
                                                 yval=yval_bit,
                                                 save_best_only=True,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=50)


# In[6]:

cvae_model.full_model.fit([xtr, ytr_bit], [ytr_bit], batch_size=10, epochs=50, validation_data=([xval, yval_bit], yval_bit),
                   callbacks=[model_ckpt_callback])


# #### Predictions using decoder part
# 
# $ \phi_i = \mu(x_i,u_i,\theta'') $

# In[10]:

best_model = CVAE(n_hidden_units=n_u)
best_model.full_model.load_weights(cvae_best_ckpt_path)


# In[12]:

results = dict()
results['train'] = best_model.evaluate(xtr, ytr_deg, 'train')
results['validation'] = best_model.evaluate(xval, yval_deg, 'validation')
results['test'] = best_model.evaluate(xte, yte_deg, 'test')

