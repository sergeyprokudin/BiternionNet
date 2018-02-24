# Dataset itself can be downloaded here: http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz

import os
import numpy as np
from scipy.io import loadmat
from utils.angles import cart_to_spherical, rad2bit


def _load_normalized_mat(mat_path):

    mat = loadmat(mat_path)
    data = mat['data']

    left_eye = data['left']
    # right_eye = data['right']

    def _parse_eye(eye_data):
        gaze = eye_data[0][0][0][0][0]
        img = eye_data[0][0][0][0][1]
        hpose = eye_data[0][0][0][0][2]
        return gaze, img, hpose

    lgaze, limg, lhpose = _parse_eye(left_eye)
    # rgaze, rimg, rhpose = _parse_eye(right_eye)

    # gaze = np.vstack([lgaze, rgaze])
    # img = np.vstack([limg, rimg])
    # hpose = np.vstack([lhpose, rhpose])

    return lgaze, limg, lhpose


def _load_person_data(person_data_path):

    gaze_lst = []
    img_lst = []
    hpose_lst = []

    for mat_file in os.listdir(person_data_path):

        mat_path = os.path.join(person_data_path, mat_file)
        gaze, img, hpose = _load_normalized_mat(mat_path)
        gaze_lst.append(gaze)
        img_lst.append(img)
        hpose_lst.append(hpose)

    gaze = np.vstack(gaze_lst)
    img = np.vstack(img_lst)
    hpose = np.vstack(hpose_lst)

    return gaze, img, hpose


def load_dataset(normalized_data_path, validation_subject_id=13, test_subject_id=14):

    person_folders = os.listdir(normalized_data_path)

    train_data = {'gaze_xyz': [], 'img': [], 'hpose': []}
    validation_data = {'gaze_xyz': [], 'img': [], 'hpose': []}
    test_data = {'gaze_xyz': [], 'img': [], 'hpose': []}

    for pid, pf in enumerate(person_folders):

        print("loading data for person %d" % pid)

        person_data_path = os.path.join(normalized_data_path, pf)

        gaze, img, hpose = _load_person_data(person_data_path)

        if pid==validation_subject_id:
            validation_data['gaze_xyz'].append(gaze)
            validation_data['img'].append(img)
            validation_data['hpose'].append(hpose)
        elif pid==test_subject_id:
            test_data['gaze_xyz'].append(gaze)
            test_data['img'].append(img)
            test_data['hpose'].append(hpose)
        else:
            train_data['gaze_xyz'].append(gaze)
            train_data['img'].append(img)
            train_data['hpose'].append(hpose)

    def _vstack(data):
        for key in data.keys():
            data[key] = np.vstack(data[key])

    _vstack(train_data)
    _vstack(validation_data)
    _vstack(test_data)

    def _expand_img_dims(data):
        data['img'] = np.expand_dims(data['img'], axis=-1)

    _expand_img_dims(train_data)
    _expand_img_dims(validation_data)
    _expand_img_dims(test_data)

    def _convert_angles(data):
        gaze_spherical = cart_to_spherical(data['gaze_xyz'])
        data['yaw_rad'] = gaze_spherical[:, 1]
        data['yaw_deg'] = np.rad2deg(data['yaw_rad'])
        data['yaw_bit'] = rad2bit(data['yaw_rad'])
        data['pitch_rad'] = gaze_spherical[:, 2]
        data['pitch_deg'] = np.rad2deg(data['pitch_rad'])
        data['pitch_bit'] = rad2bit(data['pitch_rad'])

    _convert_angles(train_data)
    _convert_angles(validation_data)
    _convert_angles(test_data)

    return train_data, validation_data, test_data
