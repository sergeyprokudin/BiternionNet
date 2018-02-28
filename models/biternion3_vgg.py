from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, von_mises_log_likelihood_tf
from utils.losses import von_mises_log_likelihood_np, von_mises_neg_log_likelihood_keras

def biternion3_vgg(image_height, image_width, n_channels,
                   name='biternion_vgg', loss_type='cosine'):

    X_input = Input(shape=[image_height, image_width, n_channels], name='input_image')

    vgg_img = vgg.vgg_model(final_layer=False,
                            image_height=image_height,
                            image_width=image_width,
                            n_channels=n_channels)(X_input)

    theta_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='theta_mean')(Dense(2, activation='linear')(vgg_img))
    theta_kappa =  Lambda(lambda x: K.abs(x), name='theta_kappa')(Dense(1, activation='linear')(vgg_img))

    phi_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='phi_mean')(Dense(2, activation='linear')(vgg_img))
    phi_kappa =  Lambda(lambda x: K.abs(x), name='phi_kappa')(Dense(1, activation='linear')(vgg_img))

    psi_mean = Lambda(lambda x: K.l2_normalize(x, axis=1), name='psi_mean')(Dense(2, activation='linear')(vgg_img))
    psi_kappa =  Lambda(lambda x: K.abs(x), name='psi_kappa')(Dense(1, activation='linear')(vgg_img))

    y_pred = concatenate([theta_mean, theta_kappa, phi_mean, phi_kappa, psi_mean, psi_kappa])

    def _unpack_preds(y_pred):

        theta_mean = y_pred[:, 0:2]
        theta_kappa =  y_pred[:, 2:3]
        phi_mean = y_pred[:, 3:5]
        phi_kappa =  y_pred[:, 5:6]
        psi_mean = y_pred[:, 6:8]
        psi_kappa =  y_pred[:, 7:8]

        return theta_mean, theta_kappa, phi_mean, phi_kappa, psi_mean, psi_kappa

    def _unpack_target(y_target):

        theta_target = y_target[:, 0:2]
        phi_target = y_target[:, 2:4]
        psi_target = y_target[:, 4:6]

        return theta_target, phi_target, psi_target

    def _loss(y_target, y_pred):

        theta_mean, theta_kappa, phi_mean, phi_kappa, psi_mean, psi_kappa = _unpack_preds(y_pred)
        theta_target, phi_target, psi_target = _unpack_target(y_target)

        if loss_type=='cosine':
            theta_loss = cosine_loss_tf(theta_target, theta_mean)
            phi_loss = cosine_loss_tf(phi_target, phi_mean)
            psi_loss = cosine_loss_tf(psi_target, psi_mean)
            loss = theta_loss + phi_loss + psi_loss

        elif loss_type=='vm_likelihood':
            theta_loss = von_mises_log_likelihood_tf(theta_target, theta_mean, theta_kappa)
            phi_loss = von_mises_log_likelihood_tf(phi_target, phi_mean, phi_kappa)
            psi_loss = von_mises_log_likelihood_tf(psi_target, psi_mean, psi_kappa)
            loss = -theta_loss - phi_loss - psi_loss

        return loss

    model = Model(X_input, y_pred, name=name)

    model.compile(optimizer='Adam', loss=_loss)

    return model, _unpack_preds


def evaluate_model(model, images, y_target, data_part):

    y_pred = model.predict(images, batch_size=32, verbose=1)

    def _unpack_preds(y_pred):

        theta_mean = y_pred[:, 0:2]
        theta_kappa =  y_pred[:, 2:3]
        phi_mean = y_pred[:, 3:5]
        phi_kappa =  y_pred[:, 5:6]
        psi_mean = y_pred[:, 6:8]
        psi_kappa =  y_pred[:, 7:8]

        return theta_mean, theta_kappa, phi_mean, phi_kappa, psi_mean, psi_kappa

    def _unpack_target(y_target):

        theta_target = y_target[:, 0:2]
        phi_target = y_target[:, 2:4]
        psi_target = y_target[:, 4:6]

        return theta_target, phi_target, psi_target

    theta_target, phi_target, psi_target = _unpack_target(y_target)
    theta_mean, theta_kappa, phi_mean, phi_kappa, psi_mean, psi_kappa = unpack_func(y_pred)

    preds_theta = bit2deg(theta_mean)
    gt_theta = bit2deg(theta_target)
    theta_maad = np.mean(maad_from_deg(gt_theta, preds_theta))
    print("MAAD-azimuth (%s): %f" %(data_part, theta_maad))
    theta_ll = np.mean(von_mises_log_likelihood_np(theta_target, theta_mean, theta_kappa))
    print("Log-likelihood-azimuth (%s): %f" %(data_part, theta_ll))


    preds_phi = bit2deg(phi_mean)
    gt_phi = bit2deg(phi_target)
    phi_maad = np.mean(maad_from_deg(gt_phi, preds_phi))
    print("MAAD-elevation (%s): %f" %(data_part, phi_maad))
    phi_ll = np.mean(von_mises_log_likelihood_np(phi_target, phi_mean, phi_kappa))
    print("Log-likelihood-elevation (%s): %f" %(data_part, phi_ll))

    preds_psi = bit2deg(psi_mean)
    gt_psi = bit2deg(psi_target)
    psi_maad = np.mean(maad_from_deg(gt_psi, preds_psi))
    print("MAAD-tilt (%s): %f" %(data_part, psi_maad))
    psi_ll = np.mean(von_mises_log_likelihood_np(psi_target, psi_mean, psi_kappa))
    print("Log-likelihood-tilt (%s): %f" %(data_part, psi_ll))

    total_maad = np.mean([theta_maad, phi_maad, psi_maad])
    print("MAAD (%s): %f" %(data_part, total_maad))

    return