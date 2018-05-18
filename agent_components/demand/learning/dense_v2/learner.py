import logging
import time

from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import sgd
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
        # creating model
        model = Sequential()
        # input layer
        input_shape = (cfg.DEMAND_ONE_WEEK,)
        # model.add(Dense(1, input_shape=input_shape, kernel_regularizer= l2(0.05)))
        model.add(Dense(168,
                        input_shape=input_shape,
                        kernel_regularizer=l2(0.1)
                        ))
        model.add(Dense(100))
        #model.add(Dropout(0.2))
        model.add(Dense(50))
        #model.add(Dropout(0.2))
        model.add(Dense(24))
        start = time.time()
        optimizr = sgd(lr=0.02)
        model.compile(loss='mae', optimizer=optimizr)
        log.info('compilation time : {}'.format(time.time() - start))
        return model
