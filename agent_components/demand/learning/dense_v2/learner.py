import logging
import time

from keras.layers import BatchNormalization
from keras.layers.core import Dense, Activation
from keras.models import Sequential
from keras.optimizers import sgd, SGD
from keras.regularizers import l2
from keras.utils import Sequence
from sklearn import preprocessing

import util.config as cfg
from agent_components.demand.learning.DemandLearner import DemandLearner
from util.learning_utils import get_callbacks, get_callbacks_with_generator

log = logging.getLogger(__name__)


def get_instance(tag, fresh):
    return DenseLearner(tag, fresh)


class DenseLearner(DemandLearner):
    def __init__(self, tag, fresh):
        super().__init__("dense_v2", tag, fresh)

    def learn(self):
        self._fit_offline(True)

    def fresh_model(self):
        input_shape = (cfg.DEMAND_ONE_WEEK,)
        model = Sequential()
        model.add(Dense(168, input_shape=input_shape, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(24))
        model.add(Activation('linear'))
        opti = SGD(lr=0.0001)
        model.compile(loss='mse', optimizer='sgd')
        return model
