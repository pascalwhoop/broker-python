import logging
from unittest.mock import Mock

import numpy as np
from keras.utils import Sequence
from sklearn.metrics import mean_absolute_error

import agent_components.demand.event_listeners as el
from agent_components.demand.data import parse_usage_game_log, make_sequences_from_historical
from agent_components.demand.learning.DemandLearner import DemandLearner
from util.learning_utils import ModelWriter, TbWriterHelper, get_usage_file_paths
from util.strings import MODEL_FS_NAME

log = logging.getLogger(__name__)

model_name = "baseline"
model_fs_name = model_name

def get_instance(tag, fresh):
    return BaselineLearner(tag, fresh)

class BaselineLearner(DemandLearner):
    def __init__(self, tag, fresh):
        super().__init__("baseline", tag, fresh)

    def learn(self):
        self._fit_offline(True)

    def fresh_model(self):
        return FakeModel(self.tb_writer_helper)


class FakeModel:

    def __init__(self, writer: TbWriterHelper):
        self.writer = writer

    def save_weights(self, *args):
        pass

    def fit_generator(self, s: Sequence, **kwargs):
        for i in range(s.__len__()):
            batch_x, batch_y = s.__getitem__(i)
            batch_loss = 0
            for i in range(len(batch_x)):

                # equivalent to saying "same as 24h ago"
                #mae = mean_absolute_error(batch_y[i], batch_x[i][-24:])

                # same as saying "same as 1h ago (in last timestep before realization)
                mae = mean_absolute_error(batch_y[i], batch_x[i,-len(batch_y[i]):])

                batch_loss += mae
            batch_loss = batch_loss / len(batch_x)
            self.writer.write_train_loss(batch_loss)




