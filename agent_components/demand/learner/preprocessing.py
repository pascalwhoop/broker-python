from typing import List

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from sklearn import preprocessing
import numpy as np
import os
import pickle
import logging


BATCH_SIZE        = 20  # number of sequences to feed to the model at once and whose errors are added up before propagated
SAMPLING_RATE     = 3   # assuming correlation between hours somewhere in this range (6h ago, 12h ago, 18h ago, 24h ago,..)
SEQUENCE_LENGTH   = 168 # one week sequences because that's a probable range for patterns
DATAPOINTS_PER_TS = 17  # number of datapoints in each timestep. That's customer data, weather, usage etc
VALIDATION_PART   = 0.05

def generate_batches_for_customer(data, targets):
    """[keras docs](https://keras.io/preprocessing/sequence/#timeseriesgenerator)"""
    return TimeseriesGenerator(data,
                               targets,
                               length=SEQUENCE_LENGTH,
                               sampling_rate=SAMPLING_RATE,
                               batch_size=BATCH_SIZE)


# getting the files path of data to train on
os.getcwd()
consume_files_path = os.path.abspath(os.path.join(os.getcwd(), "data/consume"))
files = os.listdir(consume_files_path)
files.sort()


def get_game_data(train_path, label_path):
    logging.info("getting {}".format(train_path))
    with open(label_path, "rb") as fh:
        labels        = pickle.load(fh)
    with open(train_path, "rb") as fh:
        training_data = pickle.load(fh)
    return [training_data, labels]


class CustomerIterator:
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
        data                                   = preprocessing.scale(data)
        targets                                = preprocessing.scale(targets)
        cutoff                                 = int(len(data) * VALIDATION_PART) + SEQUENCE_LENGTH
        training_sequence, validation_sequence = self.make_sequences(cutoff, data, targets)
        validation_tuple = self.convert_sequence_to_set(validation_sequence)
        return [training_sequence, validation_tuple]

    def make_sequences(self, cutoff, data, targets):
        """given a cutoff and two data arrays, generate Sequences from it (one before cutoff one after"""
        training_sequence   = generate_batches_for_customer(np.array(data[:-cutoff]), np.array(targets[:-cutoff]))
        validation_sequence = generate_batches_for_customer(np.array(data[-cutoff:]), np.array(targets[-cutoff:]))
        return training_sequence, validation_sequence

    def get_next_pair(self):
        """pops the first of both arrays out and returns it"""
        data    = self.data_customers.pop(0)
        targets = self.targets_customers.pop(0)
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


class CustomSequenceDenseTraining (Sequence):
    def __init__(self, game_iter: GamesIterator, batch_size):
        self.game_iter = game_iter
        self.batch_size = batch_size
        self.x_batches = None
        self.y_batches = None
        self.make_batches()
        self.i = 0

    def __len__(self):
        return len(self.x_batches)

    def __getitem__(self, idx):
        idx = idx % self.__len__()
        batch_x = self.x_batches[idx]
        batch_y = self.y_batches[idx]
        return batch_x, batch_y

    def on_epoch_end(self):
        """adding the next game data in the memory"""
        self.make_batches()
        pass

    def make_batches(self):
        logging.info("making batches for new epoch")
        game = self.game_iter.__next__()
        #iterating over customer learning data
        for ci, customer in enumerate(game[0]):
            # iterating over customer timesteps
            for ti, ts in enumerate(customer):
                #calculate indexes of 4 important historical datapoints
                m1   = ti-1 if ti-1>=0 else ti
                m6   = ti-6 if ti-6>=0 else ti
                m24  = ti-24 if ti-24>=0 else ti
                m168 = ti-168 if ti-168>=0 else ti
                ch = game[1][ci]
                hist = [ch[m1], ch[m6], ch[m24], ch[m168]]
                ts.extend(hist)

        x = np.array(game[0])
        y = np.array(game[1])

        x1 =len(x) # count of customers
        x2 =len(x[0]) # count of ts
        x3 = len(x[0][0])

        #concatenating all customer data.
        x = x.reshape((x1*x2,x3 ))
        y = y.reshape(x1*x2)

        #cropping leftovers at the end away
        x = x[:-(len(x)%BATCH_SIZE)]
        y = y[:-(len(y)%BATCH_SIZE)]


        #reshaping into batches
        x = x.reshape(int((np.ceil(len(x)/BATCH_SIZE))), BATCH_SIZE, x3)
        y = y.reshape(int((np.ceil(len(y)/ BATCH_SIZE))), BATCH_SIZE)

        self.x_batches = x
        self.y_batches = y
        logging.info("batches done {}".format(x.shape))
