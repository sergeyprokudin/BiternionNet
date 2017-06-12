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


class VonMisesCriterion(df.Criterion):
    def __init__(self, kappa, radians=True):
        df.Criterion.__init__(self)

        self.kappa = kappa
        self.torad = 1 if radians else 0.0174532925

    def symb_forward(self, symb_in, symb_tgt):
        delta_rad = self.torad * (symb_in - symb_tgt)
        C = np.exp(2*self.kappa)
        return df.T.mean(C - df.T.exp(self.kappa * (1+df.T.cos(delta_rad))))


class Biternion(df.Module):
    def symb_forward(self, symb_in):
        return symb_in / df.T.sqrt((symb_in**2).sum(axis=1, keepdims=True))


def deg2bit(angles_deg):
    angles_rad = np.deg2rad(angles_deg)
    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


def bit2deg(angles_bit):
    return (np.rad2deg(np.arctan2(angles_bit[:,1], angles_bit[:,0])) + 360) % 360


class CosineCriterion(df.Criterion):
    def symb_forward(self, symb_in, symb_tgt):
        # For normalized `p_t_given_x` and `t`, dot-product (batched)
        # outputs a cosine value, i.e. between -1 (worst) and 1 (best)
        cos_angles = df.T.batched_dot(symb_in, symb_tgt)

        # Rescale to a cost going from 2 (worst) to 0 (best) each, then take mean.
        return df.T.mean(1 - cos_angles)


def main():
    X, y, n = pickle.load(gzip.open('data/TownCentre.pkl.gz', 'rb'))
    (Xtr, ytr, ntr), (Xte, yte, nte) = split_dataset(X, y, n, split=0.9)
    Xtr, ytr = Xtr.astype(df.floatX)/255, ytr.astype(df.floatX)
    Xte, yte = Xte.astype(df.floatX)/255, yte.astype(df.floatX)
    aug = AugmentationPipeline(Xtr, ytr, Cropper((46, 46)))
    print("Trainset: {}".format(len(Xtr)))
    print("Testset:  {}".format(len(Xte)))
    nets_linreg = [mknet_gpu(df.Linear(512, 1, initW=df.init.const(0))) for _ in range(3)]
    print('{:.3f}M params'.format(count_params(nets_linreg[0])/1000000))
    net = mknet_gpu(df.Linear(512, 1, initW=df.init.const(0)))
    trains_linreg = dotrain(net, df.MADCriterion(), aug, Xtr, ytr[:, None])
    y_preds = np.squeeze(dopred(net, aug, Xte, ensembling=ensemble_degrees, output2preds=lambda x: x, batchsize=100))
    loss = maad_from_deg(y_preds, yte)
    mean_loss = np.mean(loss)
    std_loss = np.std(loss)
    print("MAAD error (test) : %f ± %f" % (mean_loss, std_loss))
    import ipdb; ipdb.set_trace()
    return

if __name__ == '__main__':
    main()
