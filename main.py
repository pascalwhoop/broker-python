""""""
import sys
import os
import click
import logging
import logging.config
import util.make_xml_collection as mxc
import util.config as cfg
import communication.powertac_communication as comm


@click.group()
@click.option('--log-target', multiple=True, type=click.Choice(['file']))
@click.option('--log-level',  type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]))
def cli(log_target, log_level):
    """CLI interface for a python learning agent in the PowerTAC environment. Call it with the different subcommands to
    perform various activities such as learning, competing, generating data etc."""
    configure_logging(log_target, log_level)

@cli.command()
def about():
    """just prints out some text"""
    print('''
This is a broker that can compete in the powertac competition using python as a programming language
and therefore also allows for the use of GPU accelerated neural network learning technologies.
    ''')
    log.info("about info sent")

@cli.command()
@click.option('--component', type=click.Choice(cfg.AGENT_COMPONENTS))
def learn(component):
    """Triggers the learning of various components off of state files"""
    pass

@cli.command()
@click.option('--component', type=click.Choice(cfg.AGENT_COMPONENTS))
def generate_data(component):
    """Generate x/y learning data for agent components"""
    pass

@cli.command()
@click.option('--continuous', default=True)
def compete(continuous):
    """take part in a powertac competition"""
    pass


log = None
def configure_logging(log_target, log_level):
    cfg.LOG_LEVEL = log_level

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


# handling all the different options of what we want to do
if __name__ == '__main__':
    what = sys.argv[1]
    print("running command {}".format(what))

    if what == "connect":
        pass
    elif what == "demanddata":
        import agent_components.demand.make_pickled_matrix as pm
        pm.run()

 
