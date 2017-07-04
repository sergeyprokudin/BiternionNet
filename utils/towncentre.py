import numpy as np
import pickle, gzip


def split_dataset(X, y, img_names, split=0.1):
    itr, ival, ite, trs, vals, tes = [], [], [], set(), set(), set()
    for i, name in enumerate(img_names):
        # Extract the person's ID.
        pid = int(name.split('_')[1])

        # Decide where to put that person.
        if pid in trs:
            itr.append(i)
        elif pid in vals:
            ival.append(i)
        elif pid in tes:
            ite.append(i)
        else:
            rid = np.random.rand()
            if rid < 0.8:
                itr.append(i)
                trs.add(pid)
            elif (rid >= 0.8) and (rid < 0.9):
                ival.append(i)
                vals.add(pid)
            else:
                ite.append(i)
                tes.add(pid)
    return itr, ival, ite



def prepare_data(x, y):
    x, y = x.astype(np.float) / 255, y.astype(np.float)
    x = x.transpose([0, 2, 3, 1])  # [channels, height, width] -> [height, width, channels]
    # y = y.reshape(-1,1)
    return x, y


def load_towncentre(data_path,
                    canonical_split=True,
                    canonical_path="towncentre_canonical_split.pkl"):
    x, y, n = pickle.load(gzip.open(data_path, 'rb'))
    x, y = prepare_data(x, y)
    if canonical_split:
        print("using canonical datasplit..")
        with open(canonical_path, 'rb') as f:
            itr, ival, ite = pickle.load(f)
    else:
        itr, ival, ite = split_dataset(x, y, n, split=0.1)
    xtr, ytr = x[itr], y[itr]
    xval, yval = x[ival], y[ival]
    xte, yte = x[ite], y[ite]
    return xtr, ytr, xval, yval, xte, yte
