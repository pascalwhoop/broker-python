import logging
import time

from keras.layers import CuDNNGRU, regularizers
from keras.layers.core import Activation, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import rmsprop
from keras.utils import Sequence

import util.config as cfg
from util.strings import EPOCH_NO_WITH_LOSS
from util.learning_utils import AbstractLearnerInterface, ModelWriter, TbWriterHelper, GamesIterator

log = logging.getLogger(__name__)


class Learner(AbstractLearnerInterface):
    """This class' "run" method is always called by the `main.py` Click CLI based script. So when implementing new learning models,
    you need this"""
    def run(self):
        mdl = self.get_model()

        #get some helpers
        mw = ModelWriter(self.model_name)
        tb_writer = TbWriterHelper(self.model_name)
        # iterate over games (holding customer lists)
        log.info("running full learning for demand:")
        for g_number, game in enumerate(GamesIterator('demand')):
            log.info("Iterating through game number {}".format(g_number))
            #iterating over customers in game

            for epoch_n in range(cfg.DEMAND_GRU_EPOCHS_P_GAME):
                #iterating over all but last batch
                self.run_epoch(mdl, batches_x, batches_y, epoch_n, tb_writer)

    def get_model(self):
        model = Sequential()

        #input layer
        model.add(CuDNNGRU (input_shape=(cfg.DEMAND_SEQUENCE_LENGTH / cfg.DEMAND_SAMPLING_RATE,
                                         cfg.DEMAND_GRU_DATAPOINTS_PER_TS),
                            units=cfg.DEMAND_GRU_DATAPOINTS_PER_TS,
                            kernel_regularizer=regularizers.l1(0.01),
                            return_sequences=True,
                            ))
        model.add(CuDNNGRU(units=200,
                           return_sequences=True))
        model.add(Dropout(0.2))
        model.add(CuDNNGRU(units=100))
        model.add(Dropout(0.2))
        model.add(Dense(units=100))
        model.add(Dense(units=1))
        model.add(Activation('linear'))

        start = time.time()
        optimizr = rmsprop(lr=0.01)
        model.compile(loss='mae', optimizer=optimizr)
        log.info('compilation time : {}'.format(time.time() - start))
        return model



    def fit_with_generator(self, model, train_generator: Sequence, validation_set, callbacks):
        model.fit_generator(train_generator,
                            steps_per_epoch=None,  #a generator size (aka one customer) is an epoch
                            epochs=1,
                            verbose=2,  #progress bar, 2 = line per epoch
                            callbacks=callbacks,
                            validation_data=validation_set,
                            #validation_steps=None,
                            class_weight=None,
                        max_queue_size=10,
                        workers=8,
                        use_multiprocessing=True,
                        shuffle=True,
                        initial_epoch=0)

    def run_epoch(self, mdl, batches_x, batches_y, epoch_n, tb_writer):
        loss = -1
        for i in range(len(batches_x) - 1):
            loss = mdl.train_on_batch(batches_x[i], batches_y[i])
            tb_writer.write_train_loss(loss)
        test_loss = mdl.test_on_batch(batches_x[-1], batches_y[-1])
        log.info(EPOCH_NO_WITH_LOSS.format(epoch_n, test_loss))
        tb_writer.write_test_loss(test_loss)
