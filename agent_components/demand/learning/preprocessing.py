# I want to be able to have a variety of problems solved:
#
# 1. Forecasting only based on previous realised usage
# 2. Forecasting various time-distances (1-24h into the future)
# 3. Forecasting per-customer vs. per-tariff-group

# Ultimately, the agent needs to be able to forecast 1-24h for individual customers rather well
# It's also important to be able to compare the performance of the forecasting
from typing import List, Tuple

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence
import numpy as np
from sklearn import preprocessing
import logging
log = logging.getLogger(__name__)

import util.config as cfg

class DemandCustomerSequence(Sequence):

    # batches based on customers (i.e. one batch is all customers 1 timestep
    # be able to decide the "time into the future"
    # X X X X X X X X X X X - - - - - - - - - - - ?
    # EXISTING DATA         GAP                   FORECAST DESTINATION
    # allow arbitrary ? (all/none) number of features

    def __init__(self, forecast_distance, cust_features, cust_targets, flatten_sequences=False):
        """takes the y and x values.
        If x values are not provided, the training data will just contain historical usage"""
        self.forecast_distance = forecast_distance if forecast_distance > 0 else 1
        self.cust_targets = self._normalize_y(cust_targets)
        self.cust_features = None
        self.customer_count  = self.cust_targets.shape[0]
        self.timesteps_count = self.cust_targets.shape[1]
        self.features_count  = 0
        self.flatten_sequences = flatten_sequences
        #self.cust_targets = self.cust_targets.reshape(self.timesteps_count, self.customer_count)

        #in case we get also data alongside the usage realisations, we reshape them too
        if cust_features is not None:
            self.cust_features    = np.array(cust_features)
            assert len(self.cust_features[0][0]) == 17
            self.cust_features = np.array([_hotencode_times(cust) for cust in self.cust_features])
            assert len(self.cust_features[0][0]) == cfg.DEMAND_DATAPOINTS_PER_TS-1
            self.features_count  = self.cust_features.shape[2]
            #self.cust_features = self.cust_features.reshape((self.timesteps_count, self.customer_count, self.features_count))


    def __len__(self):
        """Returns the number of timesteps. In the __init__ method we reshape the array cust>ts>features into ts>cust>features"""
        #return self.cust_targets.shape[0]
        return self.cust_targets.shape[1]


    def __getitem__(self, idx):
        """
        We're getting batches here
        :param idx: index of item.
        :return: batch of customer sequences for 1 timestep
        """


        features_length = self.features_count + 1 #we add y as a feature (of the past only of course)
        batch_x = np.zeros((self.customer_count, cfg.DEMAND_SEQUENCE_LENGTH, features_length))

        #from ts till ts. rest padded with 0s (already done above)
        from_ = self._calculate_from(idx)
        beginning_offset = self._calculate_offset(idx)
        till_ = self._calculate_till(idx, beginning_offset)


        batch_y = self.cust_targets[:,idx]

        if self.cust_features is not None and till_ > 0:
            self._insert_features(batch_x, beginning_offset, from_, till_)

        if till_ > 0:
            #setting the historical y values also as x batch
                   #all_cust, all_timesteps in seq except current, last feature
            batch_x[:         ,beginning_offset:till_-from_+beginning_offset                   , -1] = self.cust_targets[:,from_:till_]

        #if flatten_sequences = True --> reshape array to flat (ready 4 dense)
        if self.flatten_sequences is True:
            batch_x = self._flatten_sequences(batch_x)

        return batch_x, batch_y

    def _insert_features(self, batch_x, beginning_offset, from_, till_):
        # all_cust, ts_from:ts_till, all_features
        features_x = self.cust_features[:, from_:till_, :]
        # create sequence of previous info
        # all cust, offset->timesteps_count, all but last feature
        batch_x[:, beginning_offset:features_x.shape[1] + beginning_offset, :-1] = features_x

    def _flatten_sequences(self, batch_x: np.array) -> np.array:
        s = batch_x.shape
        new_s = (s[0], s[1] * s[2])
        return batch_x.reshape(new_s)


    def _calculate_from(self, idx):
        from_ = idx - cfg.DEMAND_SEQUENCE_LENGTH
        if from_ < 0:
            from_ = 0
        return from_

    def _calculate_till(self, idx, offset):
        """calculate till index. Because python indexes are not inclusive, I add 1 at the end
        And because the offset might reduce the overall length (because from_ was cut off), the till_ needs to be shorter
        too."""
        till_ = idx - self.forecast_distance - offset
        if till_ < 0:
            from_ = 0
        return till_+1

    def _calculate_offset(self, idx):
        """the offset for the beginning of the timestep placement of feature sequences.
        If the demand_sequence is too long and the requested timestep position is smaller than that (i.e. we don't have
        features from timesteps smaller 0), we need to offset the beginning of the timesteps."""
        offset = idx - (cfg.DEMAND_SEQUENCE_LENGTH + self.forecast_distance)
        if offset < 0:
            offset *= -1
        else:
            offset = 0
        return offset

    def _normalize_y(self, cust_targets) -> np.array:
        return preprocessing.normalize(np.array(cust_targets).transpose()).transpose()


def drain_generator(seq: Sequence, batches=1):
    log.info("generating set from sequence of length {}".format(len(seq)))
    x = []
    y = []
    up_to = batches if batches < len(seq) else len(seq)
    for i in range(up_to):
        b = seq[i]
        x.extend(b[0])
        y.extend(b[1])
    return np.array(x), np.array(y)


def _hotencode_times(data):
    """hot-encodes the time of day / day of week data"""
    _data                 = np.array(data)
    hot_enc               = preprocessing.OneHotEncoder(sparse=False).fit_transform(_data[:,10:12])
    return np.concatenate((_data[:, 0:10], _data[:, 12:],  hot_enc), axis=1)


