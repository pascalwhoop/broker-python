import importlib
import logging

instance = None
log = logging.getLogger(__name__)

def get_instance():
    if instance is None:
        raise Exception("instance not set. set first with `configure`")
    return instance

def configure(model, tag, fresh):
    log.info("starting model loading")
    learner_module = importlib.import_module('agent_components.wholesale.learning.{}'.format(model))
    log.info("import completed")
    global instance
    instance = learner_module.get_instance(tag, fresh)