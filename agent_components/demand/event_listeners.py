import logging

import numpy as np
from pydispatch import dispatcher

from agent_components.demand import data
from communication.grpc_messages_pb2 import *
from communication.pubsub.signals import *

import agent_components.demand.learning as lrn
log = logging.getLogger(__name__)

def connect():
    # registering listeners
    dispatcher.connect(handle_timeslot_complete, signal=PB_TIMESLOT_COMPLETE, sender=dispatcher.Any)
    dispatcher.connect(handle_tariff_transaction, signal=PB_TARIFF_TRANSACTION, sender=dispatcher.Any)
    dispatcher.connect(handle_competition, signal=PB_COMPETITION, sender=dispatcher.Any)


def handle_timeslot_complete(sender, msg: PBTimeslotComplete):
    data.make_usages_for_timestep(msg)
    training_data = data.calculate_training_data(msg)

    # skipping first couple hundred timesteps
    if msg.timeslotIndex < 24*7:
        return

    x = np.array(training_data.x)
    y = np.array(training_data.y)
    loss = lrn.get_learner().train_on_batch(x, y)
    if msg.timeslotIndex % 100 is 0:
        log.info("loss: {}".format(loss))



def handle_tariff_transaction(sender, msg: PBTariffTransaction):
    data.update(msg)


def handle_competition(sender, msg: PBCompetition):
    """A new competition is started. Let's clear all the old data!"""
    data.clear()
    pass


def handle_bootstrap_data(sender, msg: PBCustomerBootstrapData):
    """
    This is a message that holds a number of usages for a customer. We take it for truth and put it into our usage sums
    """
    pass
