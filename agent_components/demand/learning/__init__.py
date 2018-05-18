import importlib

from util.learning_utils import StatefileCapableLearner
from util.strings import MODEL_NAME
from util.utils import get_now_date_file_ready

learner: StatefileCapableLearner = None


def set_learner(model_name: str, tag: str):
    """defines the currently active learner in this module"""
    module = importlib.import_module('agent_components.{}.learning.{}.learner'.format('demand', model_name))
    model_name = MODEL_NAME.format(model_name, tag, get_now_date_file_ready())
    global learner
    learner = module.Learner(model_name)
    return learner


def get_learner():
    return learner
