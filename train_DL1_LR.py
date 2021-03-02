import h5py
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Activation, Input, Dropout, LeakyReLU
#from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import *
#from keras.callbacks import Callback
from umami.preprocessing_tools import Configuration
from keras import backend as K
import umami.train_tools as utt
os.environ['KERAS_BACKEND'] = 'tensorflow'


def GetParser():
    """Argument parser for Preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocessing command line"
                                     "options.")

    parser.add_argument('-c', '--config_file', type=str,
                        required=True,
                        help="Name of the training config file")
    parser.add_argument('-e', '--epochs', default=20, type=int, help="Number\
        of trainng epochs.")
    # TODO: implementng vr_overlap
    parser.add_argument('--vr_overlap', action='store_true', help='''Option to
                        enable vr overlap removall for validation sets.''')
    parser.add_argument('-p', '--performance_check', action='store_true',
                        help="Performs performance check - can be run during"
                        " training")
    parser.add_argument('--lr_finder', action='store_true',
                        help="LR finder")
    args = parser.parse_args()
    return args


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# TODO: add gpu support

def NN_model(train_config, input_shape):
    NN_config = train_config.NN_structure
    inputs = Input(shape=input_shape)
    x = inputs
    for i, unit in enumerate(NN_config["units"]):
        x = Dense(units=unit, activation="linear",
                  kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization()(x)
        #x = LeakyReLU(alpha=0.3)(x)
        x = Activation(NN_config['activations'][i])(x)

        if "dropout_rate" in NN_config:
            x = Dropout(NN_config["dropout_rate"][i])(x)
    predictions = Dense(units=3, activation='softmax',
                        kernel_initializer='glorot_uniform')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    model_optimizer = Adam(lr=NN_config["lr"])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=model_optimizer,
        metrics=['accuracy']
    )
    return model, NN_config["batch_size"]


def TrainLargeFile(args, train_config, preprocess_config):
    print("Loading validation data (training data will be loaded per batch)")
    X_valid, Y_valid = utt.GetTestSample(
        input_file=train_config.validation_file,
        var_dict=train_config.var_dict,
        preprocess_config=preprocess_config)

    X_valid_add, Y_valid_add = None, None
    if train_config.add_validation_file is not None:
        X_valid_add, Y_valid_add = utt.GetTestSample(
            input_file=train_config.add_validation_file,
            var_dict=train_config.var_dict,
            preprocess_config=preprocess_config)
        assert X_valid.shape[1] == X_valid_add.shape[1]

    model, batch_size = NN_model(train_config, (X_valid.shape[1],))
    file = h5py.File(train_config.train_file, 'r')
    X_train = file['X_train']
    Y_train = file['Y_train']

    lr_finder = args.lr_finder
    if lr_finder:
        lr_finder = LRFinder(min_lr=1e-5, max_lr=1e-2,
                    steps_per_epoch=len(Y_train) / batch_size,
                     epochs=args.epochs)
    else:
        clr = CyclicLR(base_lr=5e-3, max_lr=1e-2,
                  step_size=(len(Y_train) / batch_size)*5, mode='exp_range', gamma=0.99994)
        # early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
        # mode='auto', baseline=None)
        early_stop = CustomStopper(monitor='val_loss', min_delta=0, patience=3, mode='auto', start_epoch=30)
    #iterations = 1517
    # step size should be about 2-10 times #iterations

    my_callback = utt.MyCallback(model_name=train_config.model_name,
                                 X_valid=X_valid,
                                 Y_valid=Y_valid,
                                 X_valid_add=X_valid_add,
                                 Y_valid_add=Y_valid_add)

    #callbacks = [reduce_lr, my_callback]
    if lr_finder:
        callbacks = [lr_finder, my_callback]
    else:
        callbacks = [clr, my_callback, early_stop]

    # create the training datasets
    # examples taken from https://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/  # noqa
    dx_train = tf.data.Dataset.from_tensor_slices(X_train)
    dy_train = tf.data.Dataset.from_tensor_slices(Y_train)
    # zip the x and y training data together and batch etc.
    train_dataset = tf.data.Dataset.zip(
        (dx_train, dy_train)).repeat().batch(batch_size)
    model.fit(x=train_dataset,
              epochs=args.epochs,
              callbacks=callbacks,
              steps_per_epoch=len(Y_train) / batch_size,
              use_multiprocessing=True,
              workers=8,
              validation_data=(X_valid, Y_valid)
              )


    plot_dir = f"{train_config.model_name}/plots"
    print("saving plots to", plot_dir)
    os.makedirs(plot_dir, exist_ok=True)
    if lr_finder:
        plt.plot(lr_finder.history['lr'], lr_finder.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.savefig(f"{plot_dir}/{args.epochs}lossLR.pdf", bbox_inches='tight', pad_inches=0.04)
        plt.savefig(f"{plot_dir}/{args.epochs}lossLR.png", bbox_inches='tight', pad_inches=0.04)
        plt.cla()
        plt.clf()
    else:
        plt.plot(clr.history['iterations'], clr.history['lr'])
        plt.xlabel('Iterations')
        plt.ylabel('Learning rate')
        plt.savefig(f"{plot_dir}/iterationsLR.pdf", bbox_inches='tight', pad_inches=0.04)
        plt.savefig(f"{plot_dir}/iterationsLR.png", bbox_inches='tight', pad_inches=0.04)
        plt.cla()
        plt.clf()

    print("Models saved:", train_config.model_name)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class LRFinder(Callback):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset.

    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5,
                                 max_lr=1e-2,
                                 steps_per_epoch=np.ceil(epoch_size/batch_size),
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])

            lr_finder.plot_loss()
        ```

    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient.

    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations
        return self.min_lr + (self.max_lr - self.min_lr) * x

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())

class CustomStopper(EarlyStopping):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                 start_epoch=100):  # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)



if __name__ == '__main__':
    args = GetParser()
    train_config = utt.Configuration(args.config_file)
    preprocess_config = Configuration(train_config.preprocess_config)

    if args.performance_check:
        utt.RunPerformanceCheck(train_config)
    else:
        TrainLargeFile(args, train_config, preprocess_config)
