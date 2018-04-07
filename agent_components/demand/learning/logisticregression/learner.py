import logging

from util.learning_utils import AbstractLearnerInterface

log = logging.getLogger(__name__)


class Learner(AbstractLearnerInterface):

    def run(self):
        pass

    def get_model(self):
        pass
