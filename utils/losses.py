import numpy as np
import tensorflow as tf
from scipy.special import i0 as bessel


def cosine_loss_np(y_target, y_pred):
    return 1 - np.sum(np.multiply(y_target, y_pred),axis=1)


def mad_loss_tf(y_target, y_pred):
    loss = tf.abs(y_target - y_pred)
    return tf.reduce_mean(loss)


def cosine_loss_tf(y_target, y_pred):
    loss = 1 - tf.reduce_sum(tf.multiply(y_target, y_pred), axis=1)
    mean_loss = tf.reduce_mean(loss, name='cosine_loss')
    return mean_loss


def von_mises_loss_np(y_target, y_pred, kappa=1):
    cosine_dist = np.sum(np.multiply(y_target, y_pred), axis=1) - 1
    vm_loss = 1 - np.exp(kappa*cosine_dist)
    return vm_loss


def von_mises_loss_tf(y_target, y_pred, kappa=1):
    cosine_dist = tf.reduce_sum(tf.multiply(y_target, y_pred), axis=1) - 1
    vm_loss = 1 - tf.exp(kappa*cosine_dist)
    mean_loss = tf.reduce_mean(vm_loss, name='von_mises_loss')
    return mean_loss

# bessel_taylor_coefs = np.asarray([1.00000000e+00, 2.50000000e-01, 1.56250000e-02,
#                                   4.34027778e-04, 6.78168403e-06, 6.78168403e-08,
#                                   4.70950280e-10, 2.40280755e-12, 9.38596699e-15,
#                                   2.89690339e-17, 7.24225848e-20, 1.49633440e-22,
#                                   2.59780277e-25, 3.84290351e-28, 4.90166264e-31,
#                                   5.44629182e-34, 5.31864436e-37, 4.60090342e-40,
#                                   3.55007980e-43, 2.45850402e-46], dtype='float64')

bessel_taylor_coefs = np.asarray([1.00000000e+00, 2.50000000e-01, 1.56250000e-02,
                                  4.34027778e-04, 6.78168403e-06], dtype='float32')


def bessel_approx_np_0(x, m=5):
    x = np.asarray(x).reshape(-1, 1)
    deg = np.arange(0, m, 1)*2
    x_tiled = np.tile(x, [1, m])
    deg_tiled = np.tile(deg, [x.shape[0],1])
    coef_tiled = np.tile(bessel_taylor_coefs[0:m].reshape(1, m), [x.shape[0], 1])
    return np.sum(np.power(x_tiled, deg_tiled)*coef_tiled, axis=1)


def bessel_approx_tf(x, m=5):
    deg = tf.reshape(tf.range(0, m, 1)*2, [1, -1])
    n_rows = tf.shape(x)[0]
    x_tiled = tf.tile(x, [1, m])
    deg_tiled = tf.tile(deg, [n_rows, 1])
    coef_tiled = tf.tile(bessel_taylor_coefs[0:m].reshape(1, m), [n_rows, 1])
    return tf.reduce_sum(tf.pow(x_tiled, tf.to_float(deg_tiled))*coef_tiled, axis=1)


def log_bessel_approx_tf(x, m=5):

    def _log_bessel_approx_0(x):
        bessel_taylor_coefs = np.asarray([1.00000000e+00, 2.50000000e-01, 1.56250000e-02,
                                  4.34027778e-04, 6.78168403e-06], dtype='float32')
        deg = tf.reshape(tf.range(0, m, 1)*2, [1, -1])
        n_rows = tf.shape(x)[0]
        x_tiled = tf.tile(x, [1, m])
        deg_tiled = tf.tile(deg, [n_rows, 1])
        coef_tiled = tf.tile(bessel_taylor_coefs[0:m].reshape(1, m), [n_rows, 1])
        val = tf.log(tf.reduce_sum(tf.pow(x_tiled, tf.to_float(deg_tiled))*coef_tiled, axis=1))
        return tf.reshape(val, [-1, 1])

    def _log_bessel_approx_large(x):
        return x - 0.5*tf.log(2*np.pi*x)

    res = tf.where(x > 5.0, _log_bessel_approx_large(x), _log_bessel_approx_0(x))

    return res


def von_mises_log_likelihood_np(y_true, mu, kappa, input_type='biternion'):
    '''
    Compute log-likelihood given data samples and predicted Von-Mises model parameters
    :param y_true: true values of an angle in biternion (cos, sin) representation
    :param mu: predicted mean values of an angle in biternion (cos, sin) representation
    :param log_kappa: predicted mean values of an angle in biternion (cos, sin) representation
    :param radian_input:
    :return:
    log_likelihood
    '''
    if input_type == 'degree':
        scaler = 0.0174533
        cosin_dist = np.cos(scaler * (y_true - mu))
    elif input_type == 'radian':
        cosin_dist = np.cos(y_true - mu)
    elif input_type == 'biternion':
        cosin_dist = np.sum(np.multiply(y_true, mu), axis=1)
    log_likelihood = kappa * cosin_dist - \
                     np.log(2 * np.pi) - np.log(bessel(kappa))
    import ipdb; ipdb.set_trace()
    return log_likelihood


def von_mises_log_likelihood_tf(y_true, mu, kappa, input_type='degree'):
    '''
    Compute log-likelihood given data samples and predicted Von-Mises model parameters
    :param y_true: true values of an angle in biternion (cos, sin) representation
    :param mu: predicted mean values of an angle in biternion (cos, sin) representation
    :param kappa: predicted kappa (inverse variance) values of an angle in biternion (cos, sin) representation
    :param radian_input:
    :return:
    log_likelihood
    '''
    # y_true = tf.to_double(y_true)
    # mu = tf.to_double(mu)
    # log_kappa = tf.to_double(log_kappa)
    if input_type == 'degree':
        scaler = 0.0174533
        cosin_dist = tf.cos(scaler * (y_true - mu))
    elif input_type == 'radian':
        cosin_dist = tf.cos(y_true - mu)
    elif input_type == 'biternion':
        cosin_dist = tf.reduce_sum(np.multiply(y_true, mu), axis=1)
    # log_likelihood = tf.exp(log_kappa) * cosin_dist - \
    #                  tf.log(2 * np.pi) + tf.log(bessel_approx_tf(tf.exp(log_kappa)))
    log_likelihood = kappa * cosin_dist - \
                     tf.log(2 * np.pi) - log_bessel_approx_tf(kappa)
    return tf.reduce_mean(log_likelihood)


def maad_from_deg(y_pred, y_target):
    return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(y_target - y_pred)), np.cos(np.deg2rad(y_target - y_pred)))))


def show_errs_deg(y_pred, y_target, epoch=-1):
    errs = maad_from_deg(y_pred, y_target)
    mean_errs = np.mean(errs, axis=1)
    std_errs = np.std(errs, axis=1)
    print("Error: {:5.2f}°±{:5.2f}°".format(np.mean(mean_errs), np.mean(std_errs)))
    print("Stdev: {:5.2f}°±{:5.2f}°".format(np.std(mean_errs), np.std(std_errs)))
