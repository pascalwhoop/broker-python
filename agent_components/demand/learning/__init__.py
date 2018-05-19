import importlib

instance = None

def get_instance():
    if instance is None:
        raise Exception("instance not set. set first with `configure`")
    return instance

def configure(model, tag, fresh):
    learner_module =  importlib.import_module('agent_components.demand.learning.{}.learner'.format(model))
    global instance
    instance = learner_module.get_instance(tag, fresh)
