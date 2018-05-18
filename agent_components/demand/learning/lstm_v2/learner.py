import logging
import time

from keras.layers import CuDNNLSTM
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

import util.config as cfg
from agent_components.demand.learning.DemandLearner import DemandLearner

log = logging.getLogger(__name__)


def get_instance(tag, fresh):
    return LstmLearner(tag, fresh)


class LstmLearner(DemandLearner):
    def __init__(self, tag, fresh, ):
        super().__init__("lstm_v2", tag, fresh)

    def learn(self):
        self._fit_offline(False)

    def fresh_model(self):
        model = Sequential()
        # input layer
        input_shape = (cfg.DEMAND_ONE_WEEK, 1)
        model.add(CuDNNLSTM(1, input_shape=input_shape, kernel_regularizer=l2(0.002)))
        # model.add(CuDNNLSTM(50))
        #model.add(Dense(100))
        #model.add(Dropout(0.2))
        #model.add(Dense(50))
        model.add(Dense(24))

        start = time.time()
        optimizr = Adam(lr=0.06)
        model.compile(loss='mae', optimizer=optimizr)
        log.info('compilation time : {}'.format(time.time() - start))
        return model
