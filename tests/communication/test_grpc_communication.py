import unittest

from pydispatch import dispatcher

import util.config as cfg
from communication.grpc_messages_pb2 import PBTimeslotComplete
from communication.powertac_communication_server import warn_about_grpc_not_implemented, GameService
from communication.pubsub import signals as sig

#class TestGrpcCommunication(unittest.TestCase):
#
#    def test_warn_about_grpc_not_implemented(self):
#        warn_about_grpc_not_implemented()


class TestGrpcCommunicationServer(unittest.TestCase):

    def test_send_grpc_message_dispatcher(self):
        #given
        gs = GameService()
        msg = PBTimeslotComplete(timeslotIndex=4)
        #catching event
        event = None
        def handle_event(sender, msg):
            nonlocal event
            event = msg

        dispatcher.connect(handle_event, signal=msg.DESCRIPTOR.name, sender=dispatcher.Any)
        gs.handlePBTimeslotComplete(msg, {})
        self.assertIsNotNone(event)
