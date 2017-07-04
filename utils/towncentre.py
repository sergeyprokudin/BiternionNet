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
                    val_test_split=0.1,
                    canonical_split=True):

    x, y, img_names = pickle.load(gzip.open(data_path, 'rb'))

    x, y = prepare_data(x, y)
    if canonical_split:
        val_test_split = 0.1
        np.random.seed(13)
    person_ids = np.asarray([int(name.split('_')[1]) for name in img_names])
    unique_pid_set = np.unique(person_ids)
    rands = np.random.rand(unique_pid_set.shape[0])

    train_pids = unique_pid_set[rands < 1-val_test_split*2]
    val_pids = unique_pid_set[(rands >= 1-val_test_split*2) & (rands < 1-val_test_split)]
    test_pids = unique_pid_set[rands > 1-val_test_split]

    ixtr = np.where(np.in1d(person_ids, train_pids))[0]
    ixval = np.where(np.in1d(person_ids, val_pids))[0]
    ixte = np.where(np.in1d(person_ids, test_pids))[0]

    xtr, ytr = x[ixtr], y[ixtr]
    xval, yval = x[ixval], y[ixval]
    xte, yte = x[ixte], y[ixte]

    import ipdb; ipdb.set_trace()

    return xtr, ytr, xval, yval, xte, yte
