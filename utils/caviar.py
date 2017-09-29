import pickle
import gzip
import numpy as np


def load_caviar(data_path,
                val_split=0.5,
                canonical_split=True,
                verbose=0):

    (xtr, ytr_deg, *_), (xvalte, yvalte_deg, *_) = pickle.load(gzip.open(data_path, 'rb'))

    # [channels, height, width] -> [height, width, channels]
    xtr = xtr.transpose([0, 2, 3, 1])
    xvalte = xvalte.transpose([0, 2, 3, 1])

    n_valtest_images = xvalte.shape[0]

    if canonical_split:
        val_split = 0.5
        np.random.seed(13)

    val_size = int(n_valtest_images * val_split)
    rix = np.random.choice(n_valtest_images, n_valtest_images, replace=False)

    np.random.seed(None)

    val_ix = rix[0:val_size]
    te_ix = rix[val_size:]

    xval = xvalte[val_ix]
    yval_deg = yvalte_deg[val_ix]

    xte = xvalte[te_ix]
    yte_deg = yvalte_deg[te_ix]

    return (xtr, ytr_deg), (xval, yval_deg), (xte, yte_deg)