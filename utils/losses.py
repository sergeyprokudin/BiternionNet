import numpy as np
import tensorflow as tf


def cosine_loss_np(y_target, y_pred):
    return 1 - np.sum(np.multiply(y_target, y_pred),axis=1)


def cosine_loss_tf(y_target, y_pred):
    loss = 1 - tf.reduce_sum(tf.multiply(y_target, y_pred),axis=1)
    mean_loss = tf.reduce_mean(loss, name='cosine_loss')
    return mean_loss


def von_mises_loss_np(y_target, y_pred, kappa=1):
    cosine_dist = np.sum(np.multiply(y_target, y_pred),axis=1) - 1
    vm_loss = 1 - np.exp(kappa*cosine_dist)
    return vm_loss


def von_mises_loss_tf(y_target, y_pred, kappa=1):
    cosine_dist = tf.reduce_sum(tf.multiply(y_target, y_pred), axis=1) - 1
    vm_loss = 1 - tf.exp(kappa*cosine_dist)
    mean_loss = tf.reduce_mean(vm_loss, name='von_mises_loss')
    return mean_loss


def maad_from_deg(preds, reals):
    return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(reals-preds)), np.cos(np.deg2rad(reals-preds)))))


def show_errs_deg(preds, reals, epoch=-1):
    errs = maad_from_deg(preds, reals)
    mean_errs = np.mean(errs, axis=1)
    std_errs = np.std(errs, axis=1)
    print("Error: {:5.2f}°±{:5.2f}°".format(np.mean(mean_errs), np.mean(std_errs)))
    print("Stdev: {:5.2f}°±{:5.2f}°".format(np.std(mean_errs), np.std(std_errs)))
