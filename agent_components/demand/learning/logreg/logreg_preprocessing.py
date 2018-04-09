from typing import List
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from sklearn import preprocessing
import numpy as np

import util.config as cfg
from agent_components.demand.learning.lstm.lstm_model import SEQUENCE_LENGTH, SAMPLING_RATE, BATCH_SIZE, VALIDATION_PART


class GruCustomerIterator:
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

        data, targets                          = self._get_next_pair()
        targets                                = preprocessing.RobustScaler().fit_transform(targets.reshape((-1,1))).flatten()
        data                                   = self._hotencode_times(data, targets)
        cutoff                                 = int(len(data) * VALIDATION_PART) + SEQUENCE_LENGTH
        training_sequence, validation_sequence = self.make_sequences(cutoff, data, targets)
        validation_tuple                       = self._convert_sequence_to_set(validation_sequence)
        return [training_sequence, validation_tuple]

    def make_sequences(self, cutoff, data, targets):
        """given a cutoff and two data arrays, generate Sequences from it (one before cutoff one after"""
        training_sequence   = self.generate_batches_for_customer(np.array(data[:-cutoff]), np.array(targets[:-cutoff]))
        validation_sequence = self.generate_batches_for_customer(np.array(data[-cutoff:]), np.array(targets[-cutoff:]))
        return training_sequence, validation_sequence

    def _get_next_pair(self):
        """pops the first of both arrays out and returns it"""
        data    = np.array(self.data_customers.pop(0))
        targets = np.array(self.targets_customers.pop(0))
        return data, targets

    def _convert_sequence_to_set(self, sequence: Sequence):
        #this is a set of tuples. We need a tuple of sets...
        x = []
        y = []
        for i in range(sequence.length):
            batch = sequence[i]
            x.extend(batch[0])
            y.extend(batch[1])
        return np.array(x), np.array(y)

    def _hotencode_times(self, data, targets):
        """hot-encodes the time of day / day of week data"""
        _data                 = np.array(data)
        assert len(_data[0]) == 17
        hot_enc               = preprocessing.OneHotEncoder(sparse=False).fit_transform(_data[:,10:12])
        return np.concatenate((_data[:, 0:10], _data[:, 12:],  hot_enc, targets.reshape(-1,1)), axis=1)


    def generate_batches_for_customer(self, data, targets):
        """[keras docs](https://keras.io/preprocessing/sequence/#timeseriesgenerator)"""
        return TimeseriesGenerator(data,
                                   targets,
                                   length=SEQUENCE_LENGTH,
                                   sampling_rate=SAMPLING_RATE,
                                   batch_size=BATCH_SIZE)








