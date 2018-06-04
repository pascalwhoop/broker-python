import logging
import random

from keras import Model
from keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler

from agent_components.demand.data import make_sequences_from_historical, parse_usage_game_log, clear
from util.learning_utils import ModelWriter, TbWriterHelper, get_callbacks_with_generator, get_usage_file_paths
from util.strings import MODEL_FS_NAME

log = logging.getLogger(__name__)


class DemandLearner:
    def __init__(self, name, tag, fresh=True):
        self.model_name = name
        self.model_fs_name = MODEL_FS_NAME.format(name, tag)
        self.model_writer = ModelWriter(self.model_fs_name, fresh=fresh)
        self.tb_writer_helper = TbWriterHelper(self.model_fs_name, fresh=fresh)
        self.model_writer.write_model_source('demand', self.model_name)
        self.scaler = None
        if fresh:
            self.model = self.fresh_model()
        else:
            self.model = self.model_writer.load_model()

    def _fit_offline(self, flat=False):
        """runs this model against offline data """
        #scaling data is possible
        for f in get_usage_file_paths():
            self._fit_on_game(f, flat)

    def _fit_on_game(self, f, flat):
        # put all usage records into memory
        log.info("learning on game {}".format(f.split("/")[-1]))
        dd, scaler = parse_usage_game_log(f)
        if self.scaler is None:
            # storing scaler as the scaler to go with during learning
            self.scaler = scaler
        sequences = make_sequences_from_historical(flat, self.scaler)
        for s in sequences:
            # before
            self.fit_generator(s)
        # when the game is finished, store the model
        clear()
        log.info("storing model to disk")
        self.model_writer.write_model(self.model)
        # TODO have to store the scaler as well.

    def fresh_model(self) -> Model:
        """Here is where you implement your model"""
        raise NotImplementedError

    def learn(self):
        """the external API"""
        raise NotImplementedError

    def fit_generator(self, s: Sequence):
        self.model.fit_generator(s, shuffle=False,use_multiprocessing=True, epochs=1, verbose=1, callbacks=get_callbacks_with_generator(self.model_fs_name), workers=8)

    def reload_model(self):
        self.model = self.model_writer.load_model()


    def predict(self, x):
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        prediction_scaled = self.model.predict(x)
        prediction = scaler.inverse_transform(prediction_scaled)
        return prediction
