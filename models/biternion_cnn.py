import tensorflow as tf
import keras
import numpy as np
from scipy import stats

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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.custom_keras_callbacks import ModelCheckpointEveryNBatch


from utils.losses import maad_from_deg
from utils.losses import cosine_loss_tf, von_mises_log_likelihood_tf
from utils.losses import von_mises_log_likelihood_np
from utils.angles import bit2deg

class BiternionCNN:

    def __init__(self,
                 input_shape=[224, 224, 3],
                 debug=False,
                 loss_type='cosine',
                 backbone_cnn='inception',
                 learning_rate=1.0e-4,
                 hlayer_size=512,
                 fixed_kappa=1.0):

        self.loss_type = loss_type
        self.input_shape = input_shape
        self.learning_rate = learning_rate

        if debug:
            x_in = Input(shape=input_shape)
            x = Flatten(input_shape=input_shape)(x_in)
            x = Dense(128, activation='relu')(x)
        else:
            if backbone_cnn=='inception':
                backbone_model = InceptionResNetV2(weights='imagenet', include_top=False,
                                                   input_shape=input_shape)
            elif backbone_cnn=='densenet':
                backbone_model = DenseNet169(weights='imagenet', include_top=False,
                                             input_shape=input_shape)

            x = backbone_model.output
            x = GlobalAveragePooling2D()(x)

        x = Dense(hlayer_size, activation='relu')(x)
        x = Dense(hlayer_size, activation='relu')(x)

        az_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='az_mean')(Dense(2, activation='linear')(Dense(128, activation='relu')(x)))
        az_kappa = Lambda(lambda x: K.abs(x), name='az_kappa')(Dense(1, activation='linear')(Dense(128, activation='relu')(x)))
        el_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='el_mean')(Dense(2, activation='linear')(Dense(128, activation='relu')(x)))
        el_kappa = Lambda(lambda x: K.abs(x), name='el_kappa')(Dense(1, activation='linear')(Dense(128, activation='relu')(x)))
        ti_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='ti_mean')(Dense(2, activation='linear')(Dense(128, activation='relu')(x)))
        ti_kappa = Lambda(lambda x: K.abs(x), name='ti_kappa')(Dense(1, activation='linear')(Dense(128, activation='relu')(x)))

        y_pred = concatenate([az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa])

        if debug:
            self.model = Model(x_in, y_pred, name='bi')
        else:
            self.model = Model(backbone_model.input, y_pred, name='BiternionInception')

        opt = Adam(lr=learning_rate)

        if loss_type == 'cosine':
            self.loss = self.cosine_loss
        elif loss_type == 'likelihood':
            self.loss = self.likelihood_loss

        self.model.compile(optimizer=opt, loss=self.loss)

    def unpack_preds(self, y_pred):

        az_mean = y_pred[:, 0:2]
        az_kappa =  y_pred[:, 2:3]

        el_mean = y_pred[:, 3:5]
        el_kappa = y_pred[:, 5:6]

        ti_mean = y_pred[:, 6:8]
        ti_kappa = y_pred[:, 8:9]

        return az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa

    def unpack_target(self, y_target):

        az_target = y_target[:, 0:2]
        el_target = y_target[:, 2:4]
        ti_target = y_target[:, 4:6]

        return az_target, el_target, ti_target

    def cosine_loss(self, y_target, y_pred):

        az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa = self.unpack_preds(y_pred)
        az_target, el_target, ti_target = self.unpack_target(y_target)

        az_loss = cosine_loss_tf(az_target, az_mean)
        el_loss = cosine_loss_tf(el_target, el_mean)
        ti_loss = cosine_loss_tf(ti_target, ti_mean)

        return az_loss + el_loss + ti_loss

    def likelihood_loss(self, y_target, y_pred):

        az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa = self.unpack_preds(y_pred)
        az_target, el_target, ti_target = self.unpack_target(y_target)

        az_loss = -von_mises_log_likelihood_tf(az_target, az_mean, az_kappa)
        el_loss = -von_mises_log_likelihood_tf(el_target, el_mean, el_kappa)
        ti_loss = -von_mises_log_likelihood_tf(ti_target, ti_mean, ti_kappa)

        return az_loss + el_loss + ti_loss

    def fit(self, x, y, validation_data, ckpt_path, epochs=1, batch_size=32, patience=5):

        early_stop_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
        model_ckpt = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=True)

        self.model.fit(x, y, validation_data=validation_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[early_stop_cb, model_ckpt])

        self.model.load_weights(ckpt_path)

    def evaluate(self, x, y_true):

        y_pred = self.model.predict(np.asarray(x))

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = self.unpack_preds(y_pred)
        az_preds_deg = bit2deg(az_preds_bit)
        el_preds_deg = bit2deg(el_preds_bit)
        ti_preds_deg = bit2deg(ti_preds_bit)

        az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)
        az_true_deg = bit2deg(az_true_bit)
        el_true_deg = bit2deg(el_true_bit)
        ti_true_deg = bit2deg(ti_true_bit)

        def maad(y_true_deg, y_pred_deg, angle='', verbose=1):
            aads = maad_from_deg(y_true_deg, y_pred_deg)
            maad = np.mean(aads)
            sem = stats.sem(aads)
            if verbose:
                print("MAAD %s : %2.2f+-%2.2fSE" % (angle, maad, sem))
            return aads, maad, sem

        az_aads, az_maad, az_sem = maad(az_true_deg, az_preds_deg, 'azimuth')
        el_aads, el_maad, el_sem = maad(el_true_deg, el_preds_deg, 'elevetion')
        ti_aads, ti_maad, ti_sem = maad(ti_true_deg, ti_preds_deg, 'tilt')

        def vm_log_likelihood(y_true_bit, y_preds_bit, kappa_preds, angle='', verbose=1):
            vm_lls = von_mises_log_likelihood_np(y_true_bit, y_preds_bit, kappa_preds)
            vm_ll_mean = np.mean(vm_lls)
            vm_ll_sem = stats.sem(vm_lls)
            if verbose:
                print("Log-likelihood %s : %2.2f+-%2.2fSE" % (angle, vm_ll_mean, vm_ll_sem))
            return vm_lls, vm_ll_mean, vm_ll_sem

        az_lls, az_ll_mean, az_ll_sem = vm_log_likelihood(az_true_bit, az_preds_bit, az_preds_kappa, 'azimuth')
        el_lls, el_ll_mean, el_ll_sem = vm_log_likelihood(el_true_bit, el_preds_bit, el_preds_kappa, 'elevation')
        ti_lls, el_ll_mean, ti_ll_sem = vm_log_likelihood(ti_true_bit, ti_preds_bit, ti_preds_kappa, 'tilt')

        lls = az_lls + el_lls + ti_lls
        ll_mean = np.mean(lls)
        ll_sem = stats.sem(lls)

        print("Log-likelihood total : %2.2f+-%2.2fSE" % (ll_mean, ll_sem))

        return

    def save_detections_for_official_eval(self, x, save_path):

        # det path example: '/home/sprokudin/RenderForCNN/view_estimation/vp_test_results/aeroplane_pred_view.txt'

        y_pred = self.model.predict(np.asarray(x))
        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = self.unpack_preds(y_pred)
        az_preds_deg = bit2deg(az_preds_bit)
        el_preds_deg = bit2deg(el_preds_bit)
        ti_preds_deg = bit2deg(ti_preds_bit)

        y_pred = np.vstack([az_preds_deg, el_preds_deg, ti_preds_deg]).T

        np.savetxt(save_path, y_pred, delimiter=' ', fmt='%i')

        return