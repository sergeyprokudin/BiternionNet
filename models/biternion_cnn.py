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
from utils.angles import bit2deg, rad2bit, bit2rad

P_UNIFORM = 0.15916927
GAMMA = 1.0e-1

class BiternionCNN:

    def __init__(self,
                 input_shape=[224, 224, 3],
                 debug=False,
                 loss_type='cosine',
                 backbone_cnn='inception',
                 learning_rate=1.0e-4,
                 hlayer_size=512,
                 fixed_kappas=[1.0, 1.0, 1.0]):

        self.loss_type = loss_type
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.az_kappa = fixed_kappas[0]
        self.el_kappa = fixed_kappas[1]
        self.ti_kappa = fixed_kappas[2]

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

        az_loss = -K.log(P_UNIFORM * GAMMA + (1 - GAMMA) * K.exp(von_mises_log_likelihood_tf(az_target, az_mean, az_kappa)))
        el_loss = -K.log(P_UNIFORM * GAMMA + (1 - GAMMA) * K.exp(von_mises_log_likelihood_tf(el_target, el_mean, el_kappa)))
        ti_loss = -K.log(P_UNIFORM * GAMMA + (1 - GAMMA) * K.exp(von_mises_log_likelihood_tf(ti_target, ti_mean, ti_kappa)))

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

    def log_likelihood(self, y_true_bit, y_preds_bit, kappa_preds, angle='', verbose=1):

        vm_lls = np.log(P_UNIFORM*GAMMA +
                        (1-GAMMA)*np.exp(von_mises_log_likelihood_np(y_true_bit, y_preds_bit, kappa_preds)))
        vm_ll_mean = np.mean(vm_lls)
        vm_ll_sem = stats.sem(vm_lls)
        if verbose:
            print("Log-likelihood %s : %2.2f+-%2.2fSE" % (angle, vm_ll_mean, vm_ll_sem))
        return vm_lls, vm_ll_mean, vm_ll_sem

    def maad(self, y_true_deg, y_pred_deg, angle='', verbose=1):
        aads = maad_from_deg(y_true_deg, y_pred_deg)
        maad = np.mean(aads)
        sem = stats.sem(aads)
        if verbose:
            print("MAAD %s : %2.2f+-%2.2fSE" % (angle, maad, sem))
        return aads, maad, sem

    def evaluate(self, x, y_true, kappas=None, verbose=1):

        y_pred = self.model.predict(np.asarray(x))

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = self.unpack_preds(y_pred)
        az_preds_deg = bit2deg(az_preds_bit)
        el_preds_deg = bit2deg(el_preds_bit)
        ti_preds_deg = bit2deg(ti_preds_bit)

        az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)
        az_true_deg = bit2deg(az_true_bit)
        el_true_deg = bit2deg(el_true_bit)
        ti_true_deg = bit2deg(ti_true_bit)

        az_aads, az_maad, az_sem = self.maad(az_true_deg, az_preds_deg, 'azimuth', verbose=verbose)
        el_aads, el_maad, el_sem = self.maad(el_true_deg, el_preds_deg, 'elevation', verbose=verbose)
        ti_aads, ti_maad, ti_sem = self.maad(ti_true_deg, ti_preds_deg, 'tilt', verbose=verbose)

        if self.loss_type == 'cosine':
            print("cosine loss, using fixed kappas..")
            az_kappa, el_kappa, ti_kappa = kappas
            az_preds_kappa = np.ones([y_true.shape[0], 1]) * az_kappa
            el_preds_kappa = np.ones([y_true.shape[0], 1]) * el_kappa
            ti_preds_kappa = np.ones([y_true.shape[0], 1]) * ti_kappa

        az_lls, az_ll_mean, az_ll_sem = self.log_likelihood(az_true_bit, az_preds_bit, az_preds_kappa,
                                                               'azimuth', verbose=verbose)
        el_lls, el_ll_mean, el_ll_sem = self.log_likelihood(el_true_bit, el_preds_bit, el_preds_kappa,
                                                               'elevation', verbose=verbose)
        ti_lls, el_ll_mean, ti_ll_sem = self.log_likelihood(ti_true_bit, ti_preds_bit, ti_preds_kappa,
                                                               'tilt', verbose=verbose)

        lls = az_lls + el_lls + ti_lls
        ll_mean = np.mean(lls)
        ll_sem = stats.sem(lls)

        maad_mean = np.mean([az_maad, el_maad, ti_maad])
        print("MAAD TOTAL: %2.2f+-%2.2fSE" % (maad_mean, az_sem))
        print("Log-likelihood TOTAL: %2.2f+-%2.2fSE" % (ll_mean, ll_sem))

        return maad_mean, ll_mean

    def save_detections_for_official_eval(self, x, save_path):

        # det path example: '/home/sprokudin/RenderForCNN/view_estimation/vp_test_results/aeroplane_pred_view.txt'

        y_pred = self.model.predict(np.asarray(x))
        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = self.unpack_preds(y_pred)
        az_preds_deg = bit2deg(az_preds_bit)
        el_preds_deg = bit2deg(el_preds_bit)
        ti_preds_deg = bit2deg(ti_preds_bit)

        y_pred = np.vstack([az_preds_deg, el_preds_deg, ti_preds_deg]).T

        np.savetxt(save_path, y_pred, delimiter=' ', fmt='%i')
        print("evaluation data saved to %s" % save_path)

        return

    def finetune_angle_kappa(self, y_true_bit, y_preds_bit, max_kappa=1000.0, verbose=1):

        kappa_vals = np.arange(0, max_kappa, 1.0)
        log_likelihoods = np.zeros(kappa_vals.shape)
        for i, kappa_val in enumerate(kappa_vals):
            kappa_preds = np.ones([y_true_bit.shape[0], 1]) * kappa_val
            log_likelihoods[i] = self.log_likelihood(y_true_bit, y_preds_bit, kappa_preds, verbose=0)[1]
            if verbose == 1:
                print("kappa: %f, log-likelihood: %f" % (kappa_val, log_likelihoods[i]))
        max_ix = np.argmax(log_likelihoods)
        fixed_kappa_value = kappa_vals[max_ix]

        if verbose:
            print("best kappa : %f" % fixed_kappa_value)

        return fixed_kappa_value

    def finetune_all_kappas(self, x, y_true, verbose=1):

        az_preds_bit, _, el_preds_bit, _, ti_preds_bit, _ = \
            self.unpack_preds(self.model.predict(np.asarray(x)))
        az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)

        print("finetuning kappas..")
        az_kappa = self.finetune_angle_kappa(az_true_bit, az_preds_bit, verbose=0)
        el_kappa = self.finetune_angle_kappa(el_true_bit, el_preds_bit, verbose=0)
        ti_kappa = self.finetune_angle_kappa(ti_true_bit, ti_preds_bit, verbose=0)

        if verbose:
            print("azimuth kappa: %f" % az_kappa)
            az_kappas = np.ones([y_true.shape[0], 1]) * az_kappa
            _, az_ll, _ = self.log_likelihood(az_true_bit, az_preds_bit, az_kappas)
            print("elevation kappa: %f" % el_kappa)
            el_kappas = np.ones([y_true.shape[0], 1]) * el_kappa
            _, az_ll, _ = self.log_likelihood(el_true_bit, el_preds_bit, el_kappas)
            print("tilt kappa: %f" % ti_kappa)
            ti_kappas = np.ones([y_true.shape[0], 1]) * ti_kappa
            _, az_ll, _ = self.log_likelihood(ti_true_bit, ti_preds_bit, ti_kappas)

        return az_kappa, el_kappa, ti_kappa

    def train_finetune_eval(self, x_train, y_train, x_val, y_val, x_test, y_test,
                            ckpt_path, batch_size=32, patience=10, epochs=200):

        self.fit(x_train, y_train, [x_val, y_val], epochs=epochs,
                 ckpt_path=ckpt_path, patience=patience, batch_size=batch_size)

        if self.loss_type == 'cosine':
            kappas = self.finetune_all_kappas(x_val, y_val, verbose=0)
        else:
            kappas = [1.0, 1.0, 1.0]

        print("EVALUATING ON TRAIN")
        train_maad, train_ll = self.evaluate(x_train, y_train, kappas)
        print("EVALUATING ON VALIDAITON")
        val_maad, val_ll = self.evaluate(x_val, y_val, kappas)
        print("EVALUATING ON TEST")
        test_maad, test_ll = self.evaluate(x_test, y_test, kappas)

        return train_maad, train_ll, val_maad, val_ll, test_maad, test_ll, kappas

    def pdf(self, x, angle='azimuth', kappa=None):

        vals = np.arange(0, 2*np.pi, 0.01)

        n_images = x.shape[0]
        x_vals_tiled = np.ones(n_images)

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_preds(self.model.predict(x))

        if angle == 'azimuth':
            mu_preds_bit = az_preds_bit
            kappa_preds = az_preds_kappa

        if self.loss_type == 'cosine':
            kappa_preds = np.ones([x.shape[0], 1]) * kappa

        pdf_vals = np.zeros([n_images, len(vals)])

        for xid, xval in enumerate(vals):
            x_bit = rad2bit(x_vals_tiled*xval)
            pdf_vals[:, xid] = P_UNIFORM*GAMMA + \
                               (1-GAMMA)*np.exp(np.squeeze(von_mises_log_likelihood_np(x_bit, mu_preds_bit, kappa_preds)))

        return vals, pdf_vals

    def plot_pdf(self, vals, pdf_vals, ax=None, target=None, predicted=None, step=1.0e-2):

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        x = np.arange(0, 2*np.pi, step)
        xticks = [0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]
        xticks_labels = ["$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels)
        ax.plot(vals, pdf_vals, label='pdf')
        # mu = np.sum(pdf_vals*vals*step)
        # ax.axvline(mu, c='blue', label='mean')
        if target is not None:
            ax.axvline(target, c='red', label='ground truth')

        if predicted is not None:
            ax.axvline(predicted, c='lightblue', label='predicted value')

        ax.set_xlim((0, 2*np.pi))
        ax.set_ylim(0, 1.0)
        ax.legend(loc=4)

        return

    def visualize_detections(self, x, y_true=None, kappa=1.0):

        import matplotlib.pyplot as plt

        n_images = x.shape[0]

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = self.unpack_preds(self.model.predict(x))
        az_preds_rad = bit2rad(az_preds_bit)

        if y_true is not None:
            az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)
            az_true_rad = bit2rad(az_true_bit)

        xvals, pdf_vals = self.pdf(x, kappa=kappa)

        for i in range(0, n_images):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(x[i])
            if y_true is not None:
                self.plot_pdf(xvals, pdf_vals[i], target=az_true_rad[i], predicted=az_preds_rad, ax=axs[1])
            else:
                self.plot_pdf(xvals, pdf_vals[i], ax=axs[1])
            fig.show()

        return