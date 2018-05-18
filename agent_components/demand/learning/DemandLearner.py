import logging

from keras import Model
from keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler

from agent_components.demand.data import make_sequences_from_historical, parse_usage_game_log
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
        if fresh:
            self.model = self.fresh_model()
        else:
            self.model = self.model_writer.load_model()

    def _fit_offline(self, flat=False):
        """runs this model against offline data """
        for f in get_usage_file_paths():
            # put all usage records into memory
            parse_usage_game_log(f)
            sequences = make_sequences_from_historical(flat)
            for s in sequences:
                self.fit_generator(s)
            #when the game is finished, store the model
            self.model_writer.write_model(self.model)

    def fresh_model(self) -> Model:
        """Here is where you implement your model"""
        raise NotImplementedError

    def learn(self):
        """the external API"""
        raise NotImplementedError

    def fit_generator(self, s: Sequence):
        self.model.fit_generator(s, shuffle=True, verbose=2, callbacks=get_callbacks_with_generator(self.model_fs_name))

    def reload_model(self):
        self.model = self.model_writer.load_model()

    def predict(self, x):
        scaler = MinMaxScaler()
        scaler.fit_transform(x)
        prediction_scaled = self.model.predict(x)
        prediction = scaler.inverse_transform(prediction_scaled)
        return prediction
