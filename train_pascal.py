
# coding: utf-8

# In[1]:


import numpy as np
import os
import h5py
import tables
from scipy.io import loadmat

from keras.preprocessing import image


# In[2]:


from keras.preprocessing import image

pascal3d_path = '/home/sprokudin/biternionnet/data/PASCAL3D+_release1.1'

def get_imagenet_data(pascal3d_path, class_name):

    
    def _load_image(img_path):

        img = image.load_img(img_path, target_size=(224, 224))
        x = np.asarray(img)

        return x 
    
    def _parse_annotation(mat_path):
    
        ann = loadmat(mat_path)

        cls_name = ann['record'][0][0][1][0][0][0]
        # bbox = ann['record'][0][0][1][0][0][1]
        viewpoint = ann['record'][0][0][1][0][0][3][0][0]

        azimuth = viewpoint[2]
        elevation = viewpoint[3]
        tilt = viewpoint[8]
        #distance =  viewpoint[4]

        #viewpoint = np.asarray([azimuth, elevation, distance])

        return azimuth, elevation
    
    #ann_path_pascal = os.path.join(pascal3d_path, 'Annotations/%s_pascal/'%class_name)
    ann_path_imagenet = os.path.join(pascal3d_path, 'Annotations/%s_imagenet/'%class_name)
    #imgs_path_pascal =os.path.join(pascal3d_path, 'Images/%s_pascal/'%class_name)
    imgs_path_imagenet = os.path.join(pascal3d_path, 'Images/%s_imagenet/'%class_name)   
    
    #pascal_annotations = os.listdir(ann_path_pascal)
    #pascal_images_fpath = [os.path.join(imgs_path_pascal, ann.split('.')[0]+'.jpg') for ann in pascal_annotations]
    #pascal_annotations_fpath = [os.path.join(ann_path_pascal, ann) for ann in pascal_annotations]

    imagenet_annotations = os.listdir(ann_path_imagenet)
    imagenet_images_fpath = [os.path.join(imgs_path_imagenet, ann.split('.')[0]+'.JPEG') for ann in imagenet_annotations]
    imagenet_annotations_fpath = [os.path.join(ann_path_imagenet, ann) for ann in imagenet_annotations]

    labels = np.zeros([len(imagenet_annotations), 3])
    images = np.zeros([len(imagenet_annotations), 224, 224, 3], dtype='uint8')

    for i in range(0, len(imagenet_annotations)):
        azimuth, elevation = _parse_annotation(imagenet_annotations_fpath[i])
        labels[i, 0] = azimuth
        labels[i, 1] = elevation
        images[i] = _load_image(imagenet_images_fpath[i])
        if i%50 == 0 :
            print("parsed %d samples.."% i)
    
    azimuth_deg =  labels[:, 0]
    elevation_deg =  labels[:, 1]
    tilt_deg =  labels[:, 2]

    data = {'images': images,
            'azimuth_deg': azimuth_deg.reshape([-1, 1]),
            'azimuth_bit': deg2bit(azimuth_deg),
            'elevation_deg': elevation_deg.reshape([-1, 1]),
            'elevation_bit': deg2bit(elevation_deg),
            'tilt_deg': tilt_deg.reshape([-1, 1]),
            'tilt_bit': deg2bit(tilt_deg)}
    
    return data


# In[3]:


from utils.angles import bit2deg, deg2bit
imagenet_train_data = get_imagenet_data(pascal3d_path, 'aeroplane')


# In[4]:


from utils.angles import bit2deg, deg2bit

def stitch_data_dicts(real_dict, syn_dict):

    stitched = {}
    for key in real_dict.keys():    
        stitched[key] = np.vstack([real_dict[key], syn_dict[key]])

    return stitched

def train_val_split(data, val_split=0.1):

    n_samples = len(data['images'])

    shuffled_samples = np.random.choice(n_samples, n_samples, replace=False)
    n_train = int((1-val_split)*n_samples)
    train_samples = shuffled_samples[0:n_train]
    val_samples = shuffled_samples[n_train:]

    train_data = {} 
    val_data = {}

    for key in data.keys():
        train_data[key] = data[key][train_samples]
        val_data[key] = data[key][val_samples]

    return train_data, val_data

def get_data(dbpath, class_name, 
             use_real=True,
             use_synthetic=True, 
             test_bboxes='gt'):
    
    pascal_db = h5py.File(dbpath, 'r')

    def _get_data_part(h5path):
        
        images = np.asarray(pascal_db[os.path.join(h5path, 'images')])
        labels = np.asarray(pascal_db[os.path.join(h5path, 'labels')])
        azimuth_deg =  labels[:, 1]
        elevation_deg =  labels[:, 2]
        tilt_deg =  labels[:, 3]

        data = {'images': images,
                'azimuth_deg': azimuth_deg.reshape([-1, 1]),
                'azimuth_bit': deg2bit(azimuth_deg),
                'elevation_deg': elevation_deg.reshape([-1, 1]),
                'elevation_bit': deg2bit(elevation_deg),
                'tilt_deg': tilt_deg.reshape([-1, 1]),
                'tilt_bit': deg2bit(tilt_deg)}
        
        return data
    
    if use_synthetic and use_real:
        train_data_real = _get_data_part(os.path.join('train_real', class_name))
        train_data_syn = _get_data_part(os.path.join('train_synthetic', class_name))
        train_data = stitch_data_dicts(train_data_real, train_data_syn)
    elif use_synthetic:
        train_data = _get_data_part(os.path.join('train_synthetic', class_name))
    elif use_real:
        train_data = _get_data_part(os.path.join('train_real', class_name))
    
    if test_bboxes == 'gt':
        test_data_real = _get_data_part(os.path.join('test_real_gt_boxes', class_name))
    else:
        test_data_real = _get_data_part(os.path.join('test_real_frcnn_boxes', class_name))
    
    # test_data_syn = _get_data_part(os.path.join('test_syn_boxes', class_name))
    
    data = {'train': train_data,
            'test': test_data_real}
    
    pascal_db.close()
    
    return train_data, test_data_real


# In[5]:


dbpath = '/home/sprokudin/biternionnet/data/pascal3d_real_synthetic100k.h5'

pascal_train_data, test_data = get_data(dbpath, 'aeroplane', use_synthetic=False)

train_val_data = stitch_data_dicts(pascal_train_data, imagenet_train_data)

train_real_data, val_data = train_val_split(train_val_data, val_split=0.2)

#train_syn_data, _ = get_data(dbpath, 'aeroplane', use_real=False)
#train_data = stitch_data_dicts(train_real_data, train_syn_data)
train_data = train_real_data

n_train_images, image_height, image_width, n_channels = train_data['images'].shape


# In[6]:

#plt.imshow(np.asarray(train_data['images'][81], dtype='uint8'))


# In[7]:


import tensorflow as tf
import keras
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet169
from keras.applications.resnet50 import ResNet50

from models import vgg

from utils.losses import maad_from_deg
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, von_mises_log_likelihood_tf
from utils.losses import von_mises_log_likelihood_np, von_mises_neg_log_likelihood_keras


def biternion_vgg(image_height, image_width, n_channels, 
                  name='inception_resnet', loss_type='cosine',
                  backbone_cnn_type='densenet',
                  learning_rate=1.0e-3, fixed_kappa=1.0):

    #X_input = Input(shape=[image_height, image_width, n_channels], name='input_image')
    
    if backbone_cnn_type=='inception_resnet':
        backbone_model = InceptionResNetV2(weights='imagenet', include_top=False, 
                                      input_shape=[image_height, image_width, n_channels])
        x = backbone_model.output
        x = GlobalAveragePooling2D()(x)
        
    elif backbone_cnn_type=='densenet':
        backbone_model = DenseNet169(weights='imagenet', include_top=False,
                              input_shape=[image_height, image_width, n_channels])
        x = backbone_model.output
        x = GlobalAveragePooling2D()(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    #     vgg_img = vgg.vgg_model(final_layer=False,
    #                             image_height=image_height,
    #                             image_width=image_width,
    #                             n_channels=n_channels)(X_input)

    theta_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='theta_mean')(Dense(2, activation='linear')(x))
    
    theta_kappa =  Lambda(lambda x: K.abs(x), name='theta_kappa')(Dense(1, activation='linear')(x))
        
    y_pred = concatenate([theta_mean, theta_kappa])
    
    def _unpack_preds(y_pred):
        
        theta_mean = y_pred[:, 0:2]
        theta_kappa =  y_pred[:, 2:3]

        return theta_mean, theta_kappa
    
    def _unpack_target(y_target):
        
        theta_target = y_target[:, 0:2]
        
        return theta_target
    
    def _loss(y_target, y_pred):
        
        theta_mean, theta_kappa = _unpack_preds(y_pred)
        theta_target = _unpack_target(y_target)
        
        if loss_type=='cosine':
            loss = cosine_loss_tf(theta_target, theta_mean)
            
        elif loss_type=='likelihood':
            loss = -von_mises_log_likelihood_tf(theta_target, theta_mean, theta_kappa)
             
        return loss
    
    model = Model(backbone_model.input, y_pred, name=name)
    
    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=_loss)
    
    return model

def finetune_kappa(model, x, y_bit, max_kappa=10.0, step=0.1, verbose=False):
    ytr_preds_bit = model.predict(x)[:, 0:2]
    kappa_vals = np.arange(0, max_kappa, step)
    log_likelihoods = np.zeros(kappa_vals.shape)
    for i, kappa_val in enumerate(kappa_vals):
        kappa_preds = np.ones([x.shape[0], 1]) * kappa_val
        log_likelihoods[i] = np.mean(von_mises_log_likelihood_np(y_bit, ytr_preds_bit, kappa_preds))
        if verbose:
            print("kappa: %f, log-likelihood: %f" % (kappa_val, log_likelihoods[i]))
    max_ix = np.argmax(log_likelihoods)
    fixed_kappa_value = kappa_vals[max_ix]
    print("best kappa : %f" % fixed_kappa_value)
    return fixed_kappa_value

from scipy import stats
from utils.losses import maad_from_deg

def evaluate_model(model, images, y_target, data_part, loss_type='cosine', fixed_kappa=None):
    
    y_pred = model.predict(images, batch_size=32, verbose=1)
    
    def _unpack_preds(y_pred):
        
        theta_mean = y_pred[:, 0:2]
        theta_kappa =  y_pred[:, 2:3]
        
        return theta_mean, theta_kappa
    
    def _unpack_target(y_target):
        
        theta_target = y_target[:, 0:2]
        
        return theta_target
    
    theta_target = _unpack_target(y_target)
    theta_mean, theta_kappa = _unpack_preds(y_pred) 
    
    preds_theta = bit2deg(theta_mean)
    gt_theta = bit2deg(theta_target)
    aads = maad_from_deg(gt_theta, preds_theta)
    theta_maad = np.mean(aads)
    theta_maad_sem = stats.sem(aads)
    print("MAAD (%s): %f+-%f" %(data_part, theta_maad, theta_maad_sem))
    if fixed_kappa is not None:
        theta_kappa = np.ones(theta_kappa.shape)*fixed_kappa
    theta_lls = von_mises_log_likelihood_np(theta_target, theta_mean, theta_kappa)
    theta_ll = np.mean(theta_lls)
    theta_ll_sem = stats.sem(theta_lls)
    print("Log-likelihood (%s): %f+-%f" %(data_part, theta_ll, theta_ll_sem))
    
    return


# In[9]:


from keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(angle='azimuth', loss='cosine', n_epochs=10, lr=1.0e-3, batch_size=32):
    
    print("defining the model..")
    
    angle_key = angle+'_bit'
    
    model = biternion_vgg(image_height, image_width, n_channels, 
                          loss_type=loss, learning_rate=lr)
    
    ckpt_path = '/home/sprokudin/biternionnet/logs/%s_model_%s.ckpt' %(angle, loss)
    
    model.save_weights(ckpt_path)
    
    early_stop_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

    model_ckpt = keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)

    print("training on %s angle, loss type: %s" % (angle.upper(), loss.upper()) )
    model.fit(train_data['images'], train_data[angle_key],
              validation_data=[val_data['images'], val_data[angle_key]],
              epochs=n_epochs,
              batch_size=batch_size,
              callbacks=[early_stop_cb, model_ckpt])

    model.load_weights(ckpt_path)
    
    if loss=='cosine':
        print("finetuning kappa value..")
        fixed_kappa = finetune_kappa(model=model, 
                                     x=val_data['images'], 
                                     y_bit=val_data[angle_key], step=0.1, verbose=False)
    else:
        fixed_kappa = None
        
    print("training finished. Checkpoint path : %s" % ckpt_path)
    
    print("evaluating %s model.." % angle.upper())
    evaluate_model(model, train_data['images'], train_data[angle_key], 'train', fixed_kappa=fixed_kappa)
    evaluate_model(model, val_data['images'], val_data[angle_key], 'validation', fixed_kappa=fixed_kappa)
    evaluate_model(model, test_data['images'], test_data[angle_key], 'test', fixed_kappa=fixed_kappa)
    
    K.clear_session()
    
    return model, fixed_kappa


# In[163]:


for i in range(0, 5):

    print("TRAINING ON COSINE!!!")
    K.clear_session()
    cosine_model, fixed_kappa = train_model(angle='azimuth', loss='cosine', n_epochs=200, lr=1.0e-4, batch_size=32)


# In[10]:


for i in range(0, 5):

    print("TRAINING ON LIKELIHOOD!!!")
    K.clear_session()
    model_likelihood, _ = train_model(angle='azimuth', loss='likelihood', n_epochs=200, lr=1.0e-4, batch_size=32)

