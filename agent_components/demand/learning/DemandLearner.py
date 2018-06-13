import logging
import random

from keras import Model

from agent_components.demand.learning.data import make_sequences_from_historical, parse_usage_game_log, clear
from util.learning_utils import ModelWriter, TbWriterHelper, get_callbacks_with_generator, get_usage_file_paths
from util.strings import MODEL_FS_NAME

log = logging.getLogger(__name__)


class DemandLearner:
    def __init__(self, name, tag, fresh=True):
        self.model_name = name
        self.model_fs_name = MODEL_FS_NAME.format(name, tag)
        self.model_writer = ModelWriter(self.model_fs_name, fresh=fresh)
        self.tb_writer_helper = TbWriterHelper(self.model_fs_name, fresh=fresh)
        self.model_writer.write_model_source('demand', self.model_fs_name)
        self.cbs = get_callbacks_with_generator(self.model_fs_name)

        #finally loading the model or creating a fresh one
        if fresh:
            self.model = self.fresh_model()
        else:
            self.model = self.reload_model()

    def _fit_offline(self, flat=False):
        """runs this model against offline data"""
        #scaling data is possible
        for f in get_usage_file_paths()[0:5]:
            self._fit_on_game(f, flat)

    def _fit_on_game(self, f, flat):
        # put all usage records into memory
        log.info("learning on game {}".format(f.split("/")[-1]))
        parse_usage_game_log(f)
        sequences = make_sequences_from_historical(flat)
        #             same as customers * size per epoch below * epoch count
        steps_for_game = len(sequences) * len(sequences[0])//5 * 1
        log.info("training on game {} for {} steps".format(f, steps_for_game))
        for step in range(steps_for_game):
            s = random.choice(sequences)
            self.model.fit_generator(s, verbose=1, epochs=1, callbacks=self.cbs, workers=8, steps_per_epoch=5)
        # when the game is finished, store the model
        clear()
        log.info("storing model to disk")
        self.model_writer.write_model(self.model)


    def fresh_model(self) -> Model:
        """Here is where you implement your model"""
        raise NotImplementedError

    def learn(self):
        """the external API"""
        raise NotImplementedError

    def reload_model(self) -> Model:
        return self.model_writer.load_model()


    def predict(self, x):
        return self.model.predict(x)
