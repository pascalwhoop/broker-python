
import logging

import numpy as np
from pydispatch import dispatcher

from communication.grpc_messages_pb2 import *
from communication.pubsub.signals import *

import agent_components.demand.learning as lrn
log = logging.getLogger(__name__)

#


def connect():
    pass

def handle_orderbook(sender, msg: PBOrderbook):
    pass
    #TODO

def handle_market_position(sender, msg: PBMarketPosition):
    pass
    #TODO

def handle_market_transaction(sender, msg: PBMarketTransaction):
    pass
    #TODO

def handle_timeslot(sender, msg: PBTimeslot):
    pass
    #TODO
    
    
