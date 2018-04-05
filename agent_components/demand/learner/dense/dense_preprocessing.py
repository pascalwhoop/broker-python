from typing import List

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
from sklearn import preprocessing
import numpy as np
import os
import pickle
import logging

from sklearn.preprocessing import RobustScaler

from agent_components.demand.learner.dense.deep_dense_demand_estimator import SEQUENCE_LENGTH, SAMPLING_RATE, BATCH_SIZE, VALIDATION_PART


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

        # normalizing all data
        x = RobustScaler().fit_transform(x)
        y = RobustScaler().fit_transform(y)


        #reshaping into batches
        x = x.reshape(int((np.ceil(len(x)/BATCH_SIZE))), BATCH_SIZE, x3)
        y = y.reshape(int((np.ceil(len(y)/ BATCH_SIZE))), BATCH_SIZE)

        self.x_batches = x
        self.y_batches = y
        logging.info("batches done {}".format(x.shape))