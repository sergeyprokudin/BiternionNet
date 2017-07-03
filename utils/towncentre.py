import numpy as np
import pickle, gzip


def split_dataset(X, y, n, canonical_split=True, split=0.9):
    if canonical_split:
        np.random.seed(0)
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
            rid = np.random.rand()
            print(rid)
            if rid < split:
                itr.append(i)
                trs.add(pid)
            else:
                ite.append(i)
                tes.add(pid)
    return (X[itr], y[itr], [n[i] for i in itr]), (X[ite], y[ite], [n[i] for i in ite])


def prepare_data(x, y):
    x, y = x.astype(np.float) / 255, y.astype(np.float)
    x = x.transpose([0, 2, 3, 1])  # [channels, height, width] -> [height, width, channels]
    # y = y.reshape(-1,1)
    return x, y


def load_towncentre(data_path, canonical_split=True):
    x, y, n = pickle.load(gzip.open(data_path, 'rb'))
    x, y = prepare_data(x, y)
    print('************splitting trval-test************')
    (xtrval, ytrval, ntrval), (xte, yte, nte) = split_dataset(x, y, n, split=0.9,
                                                              canonical_split=canonical_split)
    print('************splitting train-val************')
    (xtr, ytr, ntr), (xval, yval, nval) = split_dataset(xtrval, ytrval, ntrval, split=0.9,
                                                        canonical_split=canonical_split)
    return xtr, ytr, xval, yval, xte, yte
