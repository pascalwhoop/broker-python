"""The main entrance file for the agent. Here, commands and parameters can be defined according to the [click api](click.pocoo.org)"""
import time

import click
# !/usr/bin/env python
import importlib
import logging
import logging.config
import os

# import util.make_xml_collection as mxc
# import communication.powertac_communication as comm
import util.config as cfg
from agent_components.demand.estimator import Estimator
from agent_components.tariffs.publisher import TariffPublisher
from agent_components.wholesale.environments.LogEnvManagerAdapter import LogEnvManagerAdapter
from agent_components.wholesale.environments.WholesaleEnvironmentManager import WholesaleEnvironmentManager
from agent_components.wholesale.learning import reward_functions
from agent_components.wholesale.learning.baseline import BaselineTrader
from agent_components.wholesale.learning.tensorforce import TensorforceAgent
from communication import messages_cache
from util.learning_utils import ModelWriter



@click.group()
@click.option('--log-target', multiple=True, type=click.Choice(['file']))
@click.option('--log-level', type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]))
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
    component_configurator = get_learner_config(component)
    component_configurator.configure(model, tag, True)
    instance = component_configurator.get_instance()
    instance.learn()


@cli.command()
@click.option('--agent-type',    default='vpg',                               help="The type of agent that is to be trained. For example dqn, ppo, trpo, etc")
@click.option('--network',       default='mlp2_normalized_network',           help="What network configuration to use for this agent")
@click.option('--action-type',   default='discrete',                          help="What kind of action the agent decies upon. ")
@click.option('--preprocessing', default='simple',                            help="pass the input through a defined preprocessing function")
@click.option('--reward',        default='step_close_relative_mprice', help="define the reward function to use")
@click.option('--tag',                                                        help="add a tag to the learning session to keep track of them easier")
def wholesale(agent_type, network, action_type, preprocessing, reward, tag):
    """CLI option to train the wholesale component of the broker. It trains the broker offline, based on historical data """
    agent = TensorforceAgent(agent_type, network, action_type, preprocessing, reward, tag)
    reward_function = reward_functions.__dict__[reward]
    offline_adapter = LogEnvManagerAdapter(agent, reward_function)
    offline_adapter.subscribe()
    offline_adapter.start()
    agent.save_model()


def get_learner_config(component):
    return importlib.import_module('agent_components.{}.learning'.format(component))


#@cli.command()
#@click.option('--component', type=click.Choice(cfg.AGENT_COMPONENTS))
#def generate_data(component):
#    """Generate x/y learning data for agent components"""
#    if component == cfg.AGENT_COMPONENTS[0]:  # demand
#        import agent_components.demand.generate_data_v1.make_pickled_matrix as mpm
#        mpm.run()
#    if component == cfg.AGENT_COMPONENTS[2]:
#        raise NotImplementedError


@cli.command()
@click.option('--continuous', default=True)
@click.option('--demand-model', help="name of the model to apply to the demand predictor")
@click.option('--wholesale-model', help="name of the model to apply to the wholesale learner")
def compete(continuous, demand_model, wholesale_model):
    """take part in a powertac competition"""

    # bootstrapping logging and caching of messages
    messages_cache.subscribe()

    # bootstrapping models from stored data
    model = ModelWriter(demand_model, False).load_model()
    estimator = Estimator(model)
    estimator.subscribe()

    # TODO wholesale_trader dynamic loading
    ws_agent = BaselineTrader()
    wholesale = WholesaleEnvironmentManager(ws_agent, None)
    wholesale.subscribe()

    # simple tariff mirroring
    publisher = TariffPublisher()
    publisher.subscribe()

    # GRPC comm with powertac
    import communication.powertac_communication_server as grpc_com

    # subscribing to outgoing messages
    grpc_com.submit_service.subscribe()

    # main comm thread
    grpc_server = grpc_com.serve()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        grpc_server.stop(0)


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

    # making sure target folder exists
    if 'file' in log_target:
        print("logging to files")
        os.makedirs(cfg.LOG_PATH, exist_ok=True)

    log_cfg = cfg.get_log_config()

    # applying logging targets
    for h in log_target:
        log_cfg['handlers'][h] = cfg.get_log_handlers()[h]
        log_cfg['loggers']['']['handlers'].append(h)

    # apply logging configuration
    logging.config.dictConfig(log_cfg)

    global log
    log = logging.getLogger(__name__)
    log.info("logger configured")
    log.debug(log_cfg)


# allowing this to be called directly to let debugging work on PyCharm
script_call = click.CommandCollection(sources=[cli])
if __name__ == '__main__':
    print("calling directly")
    # configure_logging(['file'], 'DEBUG')
    cli()
    script_call()
