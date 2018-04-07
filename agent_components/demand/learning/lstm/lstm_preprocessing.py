from typing import List

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from sklearn import preprocessing
import numpy as np
import os
import pickle
import logging

from agent_components.demand.learning.lstm.lstm_model import SEQUENCE_LENGTH, SAMPLING_RATE, BATCH_SIZE, VALIDATION_PART
import util.config as cfg

def generate_batches_for_customer(data, targets):
    """[keras docs](https://keras.io/preprocessing/sequence/#timeseriesgenerator)"""
    return TimeseriesGenerator(data,
                               targets,
                               length=SEQUENCE_LENGTH,
                               sampling_rate=SAMPLING_RATE,
                               batch_size=BATCH_SIZE)


# getting the files path of data to train on
os.getcwd()
consume_files_path = os.path.join(cfg.DATA_PATH, "demand")
files = os.listdir(consume_files_path)
files.sort()


def get_game_data(train_path, label_path):
    logging.info("getting {}".format(train_path))
    with open(label_path, "rb") as fh:
        labels        = pickle.load(fh)
    with open(train_path, "rb") as fh:
        training_data = pickle.load(fh)
    return [training_data, labels]


class LSTMCustomerIterator:
    """Iterator that serves customer Sequence objects using the TimeseriesGenerator
    """
    def __init__(self, data_customers, targets_customers):
        self.data_customers    = data_customers
        self.targets_customers = targets_customers

    def __iter__(self):
        return self

    def __next__(self) -> List:
        """getting the first of the two lists and generating 1 sequence (for learning) and one tuple of data to validate
        (tensorboard doesnt like generators)
        """
        if len(self.data_customers) == 0:
            raise StopIteration

        data, targets                          = self.get_next_pair()
        targets                                = preprocessing.RobustScaler().fit_transform(targets.reshape((-1,1))).flatten()
        data                                   = self.scale_hotencode_clean(data, targets)
        cutoff                                 = int(len(data) * VALIDATION_PART) + SEQUENCE_LENGTH
        training_sequence, validation_sequence = self.make_sequences(cutoff, data, targets)
        validation_tuple                       = self.convert_sequence_to_set(validation_sequence)
        return [training_sequence, validation_tuple]

    def make_sequences(self, cutoff, data, targets):
        """given a cutoff and two data arrays, generate Sequences from it (one before cutoff one after"""
        training_sequence   = generate_batches_for_customer(np.array(data[:-cutoff]), np.array(targets[:-cutoff]))
        validation_sequence = generate_batches_for_customer(np.array(data[-cutoff:]), np.array(targets[-cutoff:]))
        return training_sequence, validation_sequence

    def get_next_pair(self):
        """pops the first of both arrays out and returns it"""
        data    = np.array(self.data_customers.pop(0))
        targets = np.array(self.targets_customers.pop(0))
        return data, targets

    def convert_sequence_to_set(self, sequence: Sequence):
        #this is a set of tuples. We need a tuple of sets...
        x = []
        y = []
        for i in range(sequence.length):
            batch = sequence[i]
            x.extend(batch[0])
            y.extend(batch[1])
        return np.array(x), np.array(y)

    def scale_hotencode_clean(self, data, targets):
        _data                 = np.array(data)
        assert len(_data[0]) == 17
        hot_enc               = preprocessing.OneHotEncoder(sparse=False).fit_transform(_data[:,10:12])
        return np.concatenate((_data[:,0:10], _data[:, 12:],  hot_enc, targets.reshape(-1,1)), axis=1)





class GamesIterator:
    """Iterator that serves games data as a 3D array [train/test, customers, timesteps]. iterating on it gives the next game and the next etc
    """
    def __init__(self):
        fp = []
        for index in range(int(len(files)/2)):
            labels_path = os.path.join(consume_files_path, files[index*2])
            training_path = os.path.join(consume_files_path, files[index*2+1])
            fp.append([training_path, labels_path])
        self.file_paths = fp

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.file_paths) == 0:
            raise StopIteration
        p = self.file_paths.pop()
        return get_game_data(p[0], p[1])



