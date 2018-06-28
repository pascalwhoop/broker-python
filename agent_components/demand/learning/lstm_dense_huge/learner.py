import logging
import time

from keras.initializers import RandomUniform
from keras.layers import CuDNNLSTM, Dropout, Activation
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

import util.config as cfg
from agent_components.demand.learning.DemandLearner import DemandLearner

log = logging.getLogger(__name__)


def get_instance(tag, fresh):
    return HugeLearner(tag, fresh)


class HugeLearner(DemandLearner):
    def __init__(self, tag, fresh, ):
        super().__init__("lstm_dense_huge", tag, fresh)

    def learn(self, games=None):
        """Meant for offline, record based learning. Not meant for competition learning"""
        self._fit_offline(False, games=games)

    def fresh_model(self):
        model = Sequential()
        # input layer
        input_shape = (cfg.DEMAND_ONE_WEEK, 1)
        model.add(CuDNNLSTM(168, input_shape=input_shape, kernel_regularizer=l2(0.002), return_sequences=True))
        model.add(CuDNNLSTM(168, input_shape=input_shape, kernel_regularizer=l2(0.002), return_sequences=True))
        model.add(CuDNNLSTM(168, input_shape=input_shape, kernel_regularizer=l2(0.002), return_sequences=False))
        model.add(Dense(500))
        model.add(Dense(200))
        model.add(Dense(24))
        model.add(Activation('linear'))

        start = time.time()
        optimizr = Adam(lr=0.01)
        model.compile(loss='mae', optimizer=optimizr)
        log.info('compilation time : {}'.format(time.time() - start))
        return model
