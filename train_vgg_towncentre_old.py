import numpy as np
import pickle, gzip
import DeepFried2 as df
from lbtoolbox.thutil import count_params
from lbtoolbox.augmentation import AugmentationPipeline, Cropper

from training_utils import dotrain, dostats, dopred
from models.vgg_theano import mknet_cpu, mknet_gpu, Flatten
from utils.towncentre import split_dataset

def ensemble_degrees(angles):
    return np.arctan2(np.mean(np.sin(np.deg2rad(angles)), axis=0), np.mean(np.cos(np.deg2rad(angles)), axis=0))


def dopred_deg(model, aug, X, batchsize=100):
    return np.rad2deg(dopred(model, aug, X, ensembling=ensemble_degrees, output2preds=lambda x: x, batchsize=batchsize))


def maad_from_deg(preds, reals):
    return np.rad2deg(np.abs(np.arctan2(np.sin(np.deg2rad(reals-preds)), np.cos(np.deg2rad(reals-preds)))))


def show_errs_deg(preds, reals, epoch=-1):
    errs = maad_from_deg(preds, reals)
    mean_errs = np.mean(errs, axis=1)
    std_errs = np.std(errs, axis=1)
    print("Error: {:5.2f}°±{:5.2f}°".format(np.mean(mean_errs), np.mean(std_errs)))
    print("Stdev: {:5.2f}°±{:5.2f}°".format(np.std(mean_errs), np.std(std_errs)))


def main():
    X, y, n = pickle.load(gzip.open('data/TownCentre.pkl.gz', 'rb'))
    (Xtr, ytr, ntr), (Xte, yte, nte) = split_dataset(X, y, n, split=0.9)
    Xtr, ytr = Xtr.astype(df.floatX)/255, ytr.astype(df.floatX)
    Xte, yte = Xte.astype(df.floatX)/255, yte.astype(df.floatX)
    aug = AugmentationPipeline(Xtr, ytr, Cropper((46,46)))
    print("Trainset: {}".format(len(Xtr)))
    print("Testset:  {}".format(len(Xte)))
    nets_shallow_linreg = [df.Sequential(Flatten(),
                                         df.Linear(3*46*46, 1, initW=df.init.const(0)),) for _ in range(1)]
    print('{:.3f}M params'.format(count_params(nets_shallow_linreg[0])/1000000))
    nets_linreg = [mknet_cpu(df.Linear(512, 1, initW=df.init.const(0))) for _ in range(1)]
    trains_linreg = [dotrain(net, df.MADCriterion(), aug, Xtr, ytr[:,None]) for net in nets_linreg]
    y_preds_linreg = [dopred_deg(net, aug, Xte) for net in nets_linreg]
    show_errs_deg(y_preds_linreg, yte[:,None])


if __name__ == '__main__':
    main()