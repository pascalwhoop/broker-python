import abc
import pickle
import numpy as np

import tensorflow as tf
import os
import logging

from keras import Model
from keras.callbacks import TensorBoard, TerminateOnNaN

from util.strings import MODEL_WEIGHTS_FILE, GETTING_GAME_X
from util.utils import deprecated

log = logging.getLogger(__name__)

import util.config as cfg


class ModelWriter:
    """Helper for Model writing. Assumes some things about the folder structure and then write the model out to disk"""

    def __init__(self, model_name):
        self.model_name = model_name
        self.storage_dir = os.path.join(cfg.MODEL_PATH, model_name)
        os.makedirs(self.storage_dir, exist_ok=True)
        self.count = 0

    def write_model(self, mdl):
        self.count += 1
        file_path = os.path.join(self.storage_dir, "game{:03d}.HDF5".format(self.count))
        mdl.save_weights(file_path)


####################################################################################


class TbWriterHelper:
    def __init__(self, model_name):
        tensorboard_dir = os.path.join(cfg.TENSORBOARD_PATH, model_name)
        self.train_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'train'))
        self.test_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'test'))
        self.train_steps = 0

    def write_train_loss(self, loss):
        summary = tf.Summary(value=[tf.Summary.Value(simple_value=loss, tag="train/loss")])
        self.train_steps += 1
        self.train_writer.add_summary(summary, global_step=self.train_steps)

    def write_test_loss(self, loss):
        summary = tf.Summary(value=[tf.Summary.Value(simple_value=loss, tag="test/loss")])
        self.test_writer.add_summary(summary, global_step=self.train_steps)


####################################################################################


class AbstractLearnerInterface:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.get_model()
        self.tb_writer_helper = TbWriterHelper(model_name)
        self.model_writer = ModelWriter(model_name)
        self.keras_callbacks = get_callbacks(model_name)

        self.split_size = cfg.VALIDATION_SPLIT

    @abc.abstractmethod
    def run(self):
        """implement this one as it should never be reached"""
        raise NotImplementedError()


    def predict(self, x):
        return self.model.predict(x)

    @abc.abstractmethod
    def get_model(self) -> Model:
        """implement this in an inheriting class and return your model"""
        raise NotImplementedError()

    def store_model(self):
        self.model_writer.write_model(self.model)

    @deprecated
    def split_game(self, game):
        x = np.array(game[0])
        y = np.array(game[1])
        ts = len(y[0])
        train_len = int((1 - self.split_size) * ts)

        train = (x[:,:train_len], y[:,:train_len])
        valid = (x[:,train_len:], y[:,train_len:])

        return train, valid

    def set_split_size(self, split):
        self.split_size =  split


####################################################################################

# standard callbacks used with keras when possible
def get_callbacks(model_name):
    return [TensorBoard(log_dir=os.path.join(cfg.TENSORBOARD_PATH, model_name),
                             histogram_freq=10,
                             batch_size=32,
                             write_grads=True,
                             write_graph=True,
                             write_images=True),
            TerminateOnNaN()]


####################################################################################


class GamesIterator:
    """
    Iterator that serves games data as a tuple (x y) with x and y being sorted into component specific terms.
    It expects a list of files under the DATA_PATH/component directory
    The files should be named as such:

    game<X><labels/training>.pickle

    The algorithm assumes this kind of naming scheme and takes every 2nd file as a training and every 1st file as a
    labels file
    """

    def __init__(self, component):
        # getting the files path of data to train on
        training_files = os.path.join(cfg.DATA_PATH, component)
        files = os.listdir(training_files)
        files.sort()

        fp = []
        for index in range(int(len(files) / 2)):
            labels_path = os.path.join(training_files, files[index * 2])
            training_path = os.path.join(training_files, files[index * 2 + 1])
            fp.append([training_path, labels_path])
        self.file_paths = fp

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.file_paths) == 0:
            raise StopIteration
        p = self.file_paths.pop()
        return self.get_game_data(p[0], p[1])

    def get_game_data(self, train_path, label_path):
        log.info(GETTING_GAME_X.format(train_path))
        with open(label_path, "rb") as fh:
            labels = pickle.load(fh)
        with open(train_path, "rb") as fh:
            training_data = pickle.load(fh)
        return training_data, labels
