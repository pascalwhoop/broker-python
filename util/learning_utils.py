import abc
import logging
import os
from typing import List

import numpy as np
import tensorflow as tf
from keras import Model
from keras.callbacks import TensorBoard, TerminateOnNaN
from shutil import rmtree

from keras.models import load_model

from util.utils import deprecated

log = logging.getLogger(__name__)

import util.config as cfg


class ModelWriter:
    """Helper for Model writing. Assumes some things about the folder structure and then write the model out to disk"""

    def __init__(self, model_name, fresh=True):
        self.model_name = model_name
        self.storage_dir = os.path.join(cfg.MODEL_PATH, model_name)
        # clearing old and overwriting
        if fresh:
            rmtree(self.storage_dir, ignore_errors=True)

        os.makedirs(self.storage_dir, exist_ok=True)
        self.count = 0

    def write_model(self, mdl):
        self.count += 1
        file_path = os.path.join(self.storage_dir, "game{:03d}.HDF5".format(self.count))
        mdl.save_weights(file_path)

    def write_model_source(self, component,name):
        import importlib, inspect
        module = importlib.import_module('agent_components.{}.learning.{}.learner'.format(component, name))
        source_code = inspect.getsource(module)
        os.makedirs(self.storage_dir, exist_ok=True)
        with open(os.path.join(self.storage_dir, "learner_code"), mode='w') as f:
            f.write(source_code)

    def load_model(self):
        return load_model(self.storage_dir)



####################################################################################


class TbWriterHelper:
    def __init__(self, model_name, fresh=True):
        tensorboard_dir = os.path.join(cfg.TENSORBOARD_PATH, model_name)
        # clearing old and overwriting
        if fresh:
            rmtree(tensorboard_dir, ignore_errors=True)
        #self.train_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'train'))
        #self.test_writer = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'test'))
        self.train_writer = tf.summary.FileWriter(tensorboard_dir)
        self.train_steps = 0

    def write_train_loss(self, loss):
        summary = tf.Summary(value=[tf.Summary.Value(simple_value=loss, tag="loss")])
        self.train_steps += 1
        self.train_writer.add_summary(summary, global_step=self.train_steps)

    def write_test_loss(self, loss):
        summary = tf.Summary(value=[tf.Summary.Value(simple_value=loss, tag="test/loss")])
        self.test_writer.add_summary(summary, global_step=self.train_steps)


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

# standard callbacks used with keras when possible
def get_callbacks_with_generator(model_name):
    return [TensorBoard(log_dir=os.path.join(cfg.TENSORBOARD_PATH, model_name),
                        batch_size=32,
                        write_grads=True,
                        write_images=True),
            TerminateOnNaN()]


####################################################################################

def get_usage_file_paths() -> List:
    files = os.listdir(cfg.DEMAND_LEARNING_USAGE_PATH)
    full_paths = []
    for f in files:
        if 'customerusage' not in f:
            continue
        full = os.path.abspath(os.path.join(cfg.DEMAND_LEARNING_USAGE_PATH, f))
        full_paths.append(full)
    return full_paths


def get_wholesale_file_paths() -> List:
    files = os.listdir(cfg.WHOLESALE_LEARNING_USAGE_PATH)
    full_paths = []
    for f in files:
        if 'marketprices' not in f:
            continue
        full = os.path.abspath(os.path.join(cfg.DEMAND_LEARNING_USAGE_PATH, f))
        full_paths.append(full)
    return full_paths
