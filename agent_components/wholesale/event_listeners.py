
import logging

import numpy as np
from pydispatch import dispatcher

from communication.grpc_messages_pb2 import *
from communication.pubsub.signals import *

import agent_components.demand.learning as lrn
log = logging.getLogger(__name__)


def connect():
    dispatcher.connect
