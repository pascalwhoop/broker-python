import asyncio
import sys
import time
from unittest.mock import patch

import grpc
import threading
import unittest
from threading import Thread

import pytest
from grpc._server import _ServerStage
from pydispatch import dispatcher

from communication.grpc_messages_pb2 import PBTimeslotComplete, PBOrder, Empty, PBSimPause, PBCompetition
from communication.powertac_communication_server import GameService, SubmitService, submit_service, serve

# def async_test(coro):
#   def wrapper(*args, **kwargs):
#       loop = asyncio.new_event_loop()
#       return loop.run_until_complete(coro(*args, **kwargs))
#   return wrapper

# class TestGrpcCommunication(unittest.TestCase):
#
#    def test_warn_about_grpc_not_implemented(self):
#        warn_about_grpc_not_implemented()
from communication.pubsub.grpc_adapter import publish_pb_message


class TestGrpcCommunicationServer(unittest.TestCase):
    def setUp(self):
        pass


    def test_send_grpc_message_dispatcher(self):
        # given
        gs = GameService()
        msg = PBTimeslotComplete(timeslotIndex=4)
        # catching event
        event = None

        def handle_event(sender, msg):
            nonlocal event
            event = msg

        dispatcher.connect(handle_event, signal=msg.DESCRIPTOR.name, sender=dispatcher.Any)
        gs.handlePBTimeslotComplete(msg, {})

        async def assertResult():
            assert event is not None

        run_in_loop(assertResult())

    def test_reverse_streaming(self):
        async def test():
            sv = SubmitService()
            print('starting test')

            # no messages at the beginning
            msgs = []

            # start the grpc thread that hooks into the generator
            async def grpc_thread():
                coro = sv.submitOrder(None, None)
                async for it in coro:
                    msgs.append(it)

            loop = asyncio.get_event_loop()
            fut = asyncio.ensure_future(grpc_thread(), loop=loop)

            # check that the length is 0
            assert len(msgs) == 0

            # then add some messages to be sent to the client
            pb_order = PBOrder(broker="jim")
            a = sv.send_order(pb_order)
            pb_order = PBOrder(broker="pip")
            b = sv.send_order(pb_order)

            async def assert_msgs_there():
                # now check again
                assert len(msgs) == 2
                assert msgs[1] == pb_order

            asyncio.ensure_future(assert_msgs_there(), loop=loop)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test())

    @patch("communication.pubsub.grpc_adapter.asyncio")
    def test_actual_grpc_reverse(self, asyncio_mock):
        """Tests the reversing of the queue async stuff with an actuall grpc client and server"""
        received = threading.Event()

        asyncio_mock.get_event_loop.return_value = asyncio.get_event_loop()

        # starting a client thread. after first message this is good enough
        def client_thread():
            import communication.grpc_messages_pb2_grpc as ptac_grpc
            channel = grpc.insecure_channel('localhost:50053')
            stub = ptac_grpc.SubmitServiceStub(channel)
            gen = stub.submitOrder(Empty())
            for i in gen:
                # received a message on the client, setting event
                received.set()
                break
            del channel
            return

        import communication.powertac_communication_server as server
        grpc_server = server.serve()

        while grpc_server._state.stage is not _ServerStage.STARTED:
            time.sleep(0.1)

        cl = threading.Thread(target=client_thread)
        cl.daemon = True
        cl.start()
        submit_service.send_order(PBOrder())
        received.wait()
        assert received.is_set()
        grpc_server.stop(1)

    def test_publish_pb_message(self):
        message = PBOrder()
        done_event = asyncio.Event()

        def listen(sender, signal, msg: PBOrder):
            assert msg == message
            done_event.set()

        dispatcher.connect(listen, signal="PBOrder")

        publish_pb_message(message)
        done_event.wait()
        # cleanup
        dispatcher.disconnect(listen, signal="PBOrder")

    def test_large_system(self):
        # start server on local host
        srvr = serve()
        import communication.grpc_messages_pb2_grpc as ptac_grpc
        channel = grpc.insecure_channel('localhost:50053')
        stub = ptac_grpc.SubmitServiceStub(channel)
        gen = stub.submitOrder(Empty())
        # sending some random stuff
        ptac_grpc.GameServiceStub(channel).handlePBSimPause(PBSimPause())
        ptac_grpc.MarketManagerServiceStub(channel).handlePBCompetition(PBCompetition())
        # after test, kill server
        srvr.stop(None)


def run_in_loop(coro_or_future):
    loop = asyncio.get_event_loop()
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    asyncio.ensure_future(coro_or_future, loop=loop)


if __name__ is "__main__":
    asyncio.get_event_loop()
