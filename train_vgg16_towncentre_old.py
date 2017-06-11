import numpy as np
import pickle, gzip
import DeepFried2 as df
from lbtoolbox.thutil import count_params
from lbtoolbox.augmentation import AugmentationPipeline, Cropper
from collections import Counter
from training_utils import dotrain, dostats, dopred

# Font which got unicode math stuff.
import matplotlib as mpl

# Much more readable plots
import matplotlib.pyplot as plt

def split(X, y, n, split=0.9):
    itr, ite, trs, tes = [], [], set(), set()
    for i, name in enumerate(n):
        # Extract the person's ID.
        pid = int(name.split('_')[1])

        # Decide where to put that person.
        if pid in trs:
            itr.append(i)
        elif pid in tes:
            ite.append(i)
        else:
            if np.random.rand() < split:
                itr.append(i)
                trs.add(pid)
            else:
                ite.append(i)
                tes.add(pid)
    return (X[itr], y[itr], [n[i] for i in itr]), (X[ite], y[ite], [n[i] for i in ite])


class Flatten(df.Module):
    def symb_forward(self, symb_in):
        return symb_in.flatten(2)


def mknet_gpu(*outlayers):
    return df.Sequential(                          #     3@46
        df.SpatialConvolutionCUDNN( 3, 24, 3, 3),  # -> 24@44
        df.BatchNormalization(24),
        df.ReLU(),
        df.SpatialConvolutionCUDNN(24, 24, 3, 3),  # -> 24@42
        df.BatchNormalization(24),
        df.SpatialMaxPoolingCUDNN(2, 2),           # -> 24@21
        df.ReLU(),
        df.SpatialConvolutionCUDNN(24, 48, 3, 3),  # -> 48@19
        df.BatchNormalization(48),
        df.ReLU(),
        df.SpatialConvolutionCUDNN(48, 48, 3, 3),  # -> 48@17
        df.BatchNormalization(48),
        df.SpatialMaxPooling(2, 2),                # -> 48@9
        df.ReLU(),
        df.SpatialConvolutionCUDNN(48, 64, 3, 3),  # -> 48@7
        df.BatchNormalization(64),
        df.ReLU(),
        df.SpatialConvolutionCUDNN(64, 64, 3, 3),  # -> 48@5
        df.BatchNormalization(64),
        df.ReLU(),
        df.Dropout(0.2),
        Flatten(),
        df.Linear(64*5*5, 512),
        df.ReLU(),
        df.Dropout(0.5),
        *outlayers
    )


def mknet_cpu(*outlayers):
    return df.Sequential(                          #     3@46
        df.SpatialConvolution( 3, 24, 3, 3),  # -> 24@44
        df.BatchNormalization(24),
        df.ReLU(),
        df.SpatialConvolution(24, 24, 3, 3),  # -> 24@42
        df.BatchNormalization(24),
        df.SpatialMaxPooling(2, 2),           # -> 24@21
        df.ReLU(),
        df.SpatialConvolution(24, 48, 3, 3),  # -> 48@19
        df.BatchNormalization(48),
        df.ReLU(),
        df.SpatialConvolution(48, 48, 3, 3),  # -> 48@17
        df.BatchNormalization(48),
        df.SpatialMaxPooling(2, 2),                # -> 48@9
        df.ReLU(),
        df.SpatialConvolution(48, 64, 3, 3),  # -> 48@7
        df.BatchNormalization(64),
        df.ReLU(),
        df.SpatialConvolution(64, 64, 3, 3),  # -> 48@5
        df.BatchNormalization(64),
        df.ReLU(),
        df.Dropout(0.2),
        Flatten(),
        df.Linear(64*5*5, 512),
        df.ReLU(),
        df.Dropout(0.5),
        *outlayers
    )

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
    (Xtr, ytr, ntr), (Xte, yte, nte) = split(X, y, n, split=0.9)
    Xtr, ytr = Xtr.astype(df.floatX)/255, ytr.astype(df.floatX)
    Xte, yte = Xte.astype(df.floatX)/255, yte.astype(df.floatX)
    aug = AugmentationPipeline(Xtr, ytr, Cropper((46,46)))
    print("Trainset: {}".format(len(Xtr)))
    print("Testset:  {}".format(len(Xte)))
    nets_shallow_linreg = [df.Sequential(
    Flatten(),
    df.Linear(3*46*46, 1, initW=df.init.const(0)),) for _ in range(1)]
    print('{:.3f}M params'.format(count_params(nets_shallow_linreg[0])/1000000))
    nets_linreg = [mknet_gpu(df.Linear(512, 1, initW=df.init.const(0))) for _ in range(1)]
    #import ipdb; ipdb.set_trace()
    trains_linreg = [dotrain(net, df.MADCriterion(), aug, Xtr, ytr[:,None]) for net in nets_linreg]
    y_preds_linreg = [dopred_deg(net, aug, Xte) for net in nets_linreg]
    show_errs_deg(y_preds_linreg, yte[:,None])


if __name__ == '__main__':
    main()