import keras
import numpy as np
import warnings

class SideModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model_name, model_to_save, save_path, save_weights_only=False):
        self.model_name = model_name
        self.model = model_to_save
        self.save_path = save_path
        self.save_weights_only = save_weights_only

    def on_train_begin(self, logs={}):
        self.epoch_id = 0
        self.min_val_loss = float("inf")

    def on_epoch_end(self, batch, logs={}):
        self.epoch_id += 1
        self.curr_val_loss = logs.get('val_loss')
        if self.curr_val_loss < self.min_val_loss:
            filepath = self.save_path.format(epoch=self.epoch_id, val_loss=self.curr_val_loss)
            print("val_loss improved from %f to %f, saving %s to %s" %
                  (self.min_val_loss, self.curr_val_loss, self.model_name, filepath))
            self.min_val_loss = self.curr_val_loss
            if self.save_weights_only:
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath)


class EvalCVAEModel(keras.callbacks.Callback):
    """ Run CVAE evaluation on selected data

    """

    def __init__(self, x, y_deg, data_title, cvae_model, ckpt_path):
        self.x = x
        self.y_deg = y_deg
        self.data_title = data_title
        self.cvae_model = cvae_model
        self.ckpt_path = ckpt_path

    def on_epoch_end(self, epoch, logs=None):
        results = self.cvae_model.evaluate_multi(self.x, self.y_deg, self.data_title)
        print("Evaluation is done.")
        if results['importance_log_likelihood'] > 0.6:
            self.model.save_weights(self.ckpt_path)
            import ipdb; ipdb.set_trace()


class ModelCheckpointEveryNBatch(keras.callbacks.Callback):
    """Save the model after every n batches, based on validation loss

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of batches) between checkpoints.
    """

    def __init__(self, filepath, xval, yval, verbose=0,
                 save_best_only=False, save_weights_only=False, period=1):
        super(ModelCheckpointEveryNBatch, self).__init__()
        self.xval = xval
        self.yval = yval
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.batches_since_last_save = 0
        self.min_val_loss = float('inf')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            filepath = self.filepath
            if self.save_best_only:
                current_loss = self.model.evaluate(self.xval, self.yval, verbose=0)
                if current_loss < self.min_val_loss:
                    if self.verbose > 0:
                        print('Batch %05d: val_loss improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (batch, self.min_val_loss,
                                 current_loss, filepath))
                    self.min_val_loss = current_loss
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('Batch %05d: val_loss did not improve' % batch)
            else:
                if self.verbose > 0:
                    print('Batch %05d: saving model to %s' % (batch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)