import tensorflow as tf
import keras

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate

from utils.losses import von_mises_log_likelihood_tf


def vgg_model(n_outputs=1, final_layer=False, l2_normalize_final=False,
              image_height=50, image_width=50):
    model = Sequential()

    model.add(Conv2D(24, kernel_size=(3, 3),
                     activation=None,
                     input_shape=[image_height, image_width, 3]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    if final_layer:
        model.add(Dense(n_outputs, activation=None))
        if l2_normalize_final:
            model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    return model


class BiternionVGG:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 predict_kappa=False,
                 fixed_kappa_value=1.0):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.predict_kappa = predict_kappa
        self.fixed_kappa_value = fixed_kappa_value

        self.X = Input(shape=[image_height, image_width, 3])

        vgg_x = vgg_model(final_layer=False,
                          image_height=self.image_height,
                          image_width=self.image_width)(self.X)

        self.y_pred = Lambda(lambda x: K.l2_normalize(x, axis=1))(Dense(2)(vgg_x))

        if predict_kappa:
            self.kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(Dense(256)(vgg_x)))
            self.model = Model(self.X, concatenate([self.y_pred, self.kappa_pred]))
        else:
            self.model = Model(self.X, self.y_pred)