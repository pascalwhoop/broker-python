#!/usr/bin/env python
import importlib
import os
import logging
import logging.config
import click

#import util.make_xml_collection as mxc
#import communication.powertac_communication as comm
import util.config as cfg
from util.strings import MODEL_NAME
from tests.util.utils import get_now_date_file_ready


@click.group()
@click.option('--log-target', multiple=True, type=click.Choice(['file']))
@click.option('--log-level',  type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]))
def cli(log_target, log_level):
    """CLI interface for a python learning agent in the PowerTAC environment. Call it with the different subcommands to
    perform various activities such as learning, competing, generating data etc."""
    configure_logging(log_target, log_level)


@cli.command()
@click.argument('component', type=click.Choice(cfg.AGENT_COMPONENTS))
@click.option('--model', help="The model of learner. It is expected to be a submodule under the component.learner. Multiple models are allowed")
@click.option('--tag', help="add a tag to your model name, allowing for easier quick expermentation and without loosing track of what was changed'")
def learn(component, model, tag):
    """Triggers the learning of various components off of state files"""
    if component in cfg.AGENT_COMPONENTS:
        module = importlib.import_module('agent_components.{}.learning.{}.learner'.format(component, model))
        model_name = MODEL_NAME.format(model, tag, get_now_date_file_ready())
        learner = module.Learner(model_name)
        log.info("Running {} learning on {}".format(model_name, component))
        learner.run()


@cli.command()
@click.option('--component', type=click.Choice(cfg.AGENT_COMPONENTS))
def generate_data(component):
    """Generate x/y learning data for agent components"""
    if component == cfg.AGENT_COMPONENTS[0]: #demand
        import agent_components.demand.generate_data.make_pickled_matrix as mpm
        mpm.run()
    if component == cfg.AGENT_COMPONENTS[2]:
        raise NotImplementedError



@cli.command()
@click.option('--continuous', default=True)
def compete(continuous):
    """take part in a powertac competition"""
    import communication.powertac_communication_server as server
    server.serve()


@cli.command()
def about():
    """just prints out some text"""
    print('''
This is a broker that can compete in the powertac competition using python as a programming language
and therefore also allows for the use of GPU accelerated neural network learning technologies.
    ''')
    log.info("about info sent")


log = None


def configure_logging(log_target, log_level):
    cfg.LOG_LEVEL = log_level if log_level else cfg.LOG_LEVEL

    #making sure target folder exists
    if 'file' in log_target:
        print("logging to files")
        os.makedirs(cfg.LOG_PATH, exist_ok=True)

    log_cfg = cfg.get_log_config()

    #applying logging targets
    for h in log_target:
        log_cfg['handlers'][h] = cfg.get_log_handlers()[h]
        log_cfg['loggers']['']['handlers'].append(h)

    #apply logging configuration
    logging.config.dictConfig(log_cfg)

    global log
    log = logging.getLogger(__name__)
    log.info("logger configured")
    log.debug(log_cfg)


#@cli.command()
#def create_sample_xml():
#    """Generates a set of sample xml files from a communication session with the server
#    """
#    comm.connect()
#    msg_counter = 0
#    completed = False
#    while not completed:
#        msg = comm.get()
#        msg_counter += 1
#        xml = mxc.parse_message(msg)
#        mxc.add_to_type_set(xml)
#        if msg_counter % 100 == 0:
#            print("Msg: {}  Known Types: {}".format(msg_counter, len(mxc.xml_types)))
#
#        if msg_counter % 1000 == 0:
#            print("pickling")
#            mxc.pickle_xml()

#allowing this to be called directly to let debugging work on PyCharm
script_call = click.CommandCollection(sources=[cli])
if __name__ == '__main__':
    print("calling directly")
    cli()
    script_call()
