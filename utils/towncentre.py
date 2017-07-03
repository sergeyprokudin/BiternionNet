import numpy as np
import pickle, gzip


def split_dataset(X, y, img_names, canonical_split=True, split=0.9):
    if canonical_split:
        np.random.seed(0)
    itr, ite, trs, tes = [], [], set(), set()
    for i, name in enumerate(img_names):
        # Extract the person's ID.
        pid = int(name.split('_')[1])

        # Decide where to put that person.
        if pid in trs:
            print('adding frame %d with person %d to test' % (i, pid))
            itr.append(i)
        elif pid in tes:
            print('adding frame %d with person %d to test' % (i, pid))
            ite.append(i)
        else:
            rid = np.random.rand()
            if rid < split:
                print("split : %f" % rid)
                print('adding frame %d with person %d to test' % (i, pid))
                itr.append(i)
                trs.add(pid)
            else:
                print("split : %f" % rid)
                print('adding frame %d with person %d to test' % (i, pid))
                ite.append(i)
                tes.add(pid)
    import ipdb; ipdb.set_trace()
    return (X[itr], y[itr], [img_names[i] for i in itr]), (X[ite], y[ite], [img_names[i] for i in ite])


def prepare_data(x, y):
    x, y = x.astype(np.float) / 255, y.astype(np.float)
    x = x.transpose([0, 2, 3, 1])  # [channels, height, width] -> [height, width, channels]
    # y = y.reshape(-1,1)
    return x, y


def load_towncentre(data_path, canonical_split=True):
    x, y, n = pickle.load(gzip.open(data_path, 'rb'))
    x, y = prepare_data(x, y)
    print(x.shape)
    print('************splitting trval-test************')
    (xtrval, ytrval, ntrval), (xte, yte, nte) = split_dataset(x, y, n, split=0.9,
                                                              canonical_split=canonical_split)
    print('************splitting train-val************')
    (xtr, ytr, ntr), (xval, yval, nval) = split_dataset(xtrval, ytrval, ntrval, split=0.9,
                                                        canonical_split=canonical_split)
    return xtr, ytr, xval, yval, xte, yte
