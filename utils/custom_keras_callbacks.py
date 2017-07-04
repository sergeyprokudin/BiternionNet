import keras


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